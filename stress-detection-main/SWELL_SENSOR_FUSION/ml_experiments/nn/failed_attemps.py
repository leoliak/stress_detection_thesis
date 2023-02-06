from time import strftime, localtime
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.svm import SVC,LinearSVC
from sklearn.linear_model import LogisticRegression as lr
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split,cross_val_score,KFold,StratifiedKFold
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2, RFE

import os, sys
import math
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
from torch.utils.data import DataLoader
#from torchvision import transforms, datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


device = torch.device("cuda:0")
print("Cuda Device Available")
print("Name of the Cuda Device: ", torch.cuda.get_device_name())
print("GPU Computational Capablity: ", torch.cuda.get_device_capability())


# def torch_train(X_tr, y_tr, X_test, y_test, input_size):
#     if torch.cuda.is_available():
#         


#     ## Model Section
#     print("Model initialization...")
#     print("Input size: {}".format(input_size))
#     net = Model(input_size, 256, 2)
#     net.to(device)

#     criterion = nn.MSELoss()
#     EPOCHS = 200
#     BATCH = 16
#     optm = torch.optim.Adam(net.parameters(), lr = 0.001)

#     dataset = {'train':stressDataset(dataset=X_tr, labels=y_tr), 'val':stressDataset(dataset=X_test, labels=y_test)}
#     loaders = {split: DataLoader(dataset[split], batch_size = batch, drop_last = True, shuffle = True) for split in ['train','val']}
    

#     def train(model, x, y, optimizer, criterion):
#         model.zero_grad()
#         output = model(x)
#         loss =criterion(output,y)
#         loss.backward()
#         optimizer.step()

#         return loss, output

#     for epoch in range(EPOCHS):
#         epoch_loss = 0
#         correct = 0
#         for bidx, batch in tqdm(enumerate(loaders['train'])):
#             x_train, y_train = batch['inp'], batch['oup']
#             x_train = x_train.view(-1,8)
#             x_train = x_train.to(device)
#             y_train = y_train.to(device)
#             loss, predictions = train(net,x_train,y_train, optm, criterion)
#             for idx, i in enumerate(predictions):
#                 i  = torch.round(i)
#                 if i == y_train[idx]:
#                     correct += 1
#             acc = (correct/len(data))
#             epoch_loss+=loss
#         print('Epoch {} Accuracy : {}'.format(epoch+1, acc*100))
#         print('Epoch {} Loss : {}'.format((epoch+1),epoch_loss))
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print('Using device:', device)






def dataset_preprocess(id_name, baseline):
    print("Load and preprocess %s data.." %id_name)
    df_name = dict_datasets[id_name]
    if not baseline:
        df_time = pd.read_csv(list_nonbaseline[df_name])
        voc = {"Condition": {"N":0, "I":1, "T":1, "R":0}}
    else:
        df_time = pd.read_csv(list_baseline[df_name])
        voc = {"Condition": {"N":0, "S":1}}

    # print(df_time.columns)
    st_scaler = preprocessing.StandardScaler()
    df_t = df_time
    df_t.replace(voc, inplace=True)
    
    labels = df_t["Condition"]
    print(labels.value_counts())
    df = df_t.drop(df_t.columns[[0,1,2,3,4]],axis=1)
    df = df.fillna(df.mean())
    df = df.replace([-np.inf], 0.0)
    feats = df.columns
    st_scaler.fit(df)
    data = st_scaler.transform(df)
    labels = labels.values

    X_tr, X_test, y_tr, y_test = train_test_split(data, labels, test_size = 0.2, random_state = 42)
    
    unique, counts = np.unique(y_tr, return_counts=True)
    print("Train set")
    print(dict(zip(unique, counts)))
    unique, counts = np.unique(y_test, return_counts=True)
    print("Test set")
    print(dict(zip(unique, counts)))
    return X_tr, X_test, y_tr, y_test, labels, feats



class stressDataset:  
    # Constructor
    def __init__(self, dataset, labels):
        # Load data
        self.data = dataset
        self.label = labels
        # Compute size
        self.size = len(self.data)
    # Get size
    def __len__(self):
        return self.size
    # Get item
    def __getitem__(self, i):
        datas = self.data[i].copy()
        targets = self.label[i].copy()
        input = datas
        classes = targets
        if targets == 0:
            target = np.array([1,0])
        else:
            target = np.array([0,1])
        return input, target, classes



class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # Call parent
        super().__init__()
        # Define parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Define internal modules
        self.layer_1 = nn.Linear(input_size, hidden_size*2)
        self.layer_2 = nn.Linear(hidden_size*2, hidden_size*4)
        self.layer_3 = nn.Linear(hidden_size*4, hidden_size*1)
        self.output = nn.Linear(hidden_size*1, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.RELU = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x1 = self.layer_1(x) #, lstm_init)
        x2 = self.RELU(x1)
        x1 = self.layer_2(x2)
        x2 = self.RELU(x1)
        x1 = self.layer_3(x2)
        x2 = self.RELU(x1)
        out = self.output(x2)
        out = self.sigmoid(out)
        return out



def train_nn_torch(X_tr, y_tr, X_test, y_test, input_size):
    batch = 64
    dataset = {'train':stressDataset(dataset=X_tr, labels=y_tr), 'val':stressDataset(dataset=X_test, labels=y_test)}
    loaders = {split: DataLoader(dataset[split], batch_size = batch, drop_last = True, shuffle = True) for split in ['train','val']}
    ###################################################################################
    
    ## Model Section
    print("Model initialization...")
    print("Input size: {}".format(input_size))
    model = Model(input_size, 256, 2)
    model.to(device)
        
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)
    criterion.to(device)

    ####################################################################################
    ## Training Section
    # Start training Session
    print("Start training process")
    tot_loss = []
    max_epoch = 200
    dist = np.zeros((max_epoch,len(X_test)))
    for epoch in range(1, max_epoch+1):
        train_correct = 0
        train_total = 0
        running_loss = 0.0

        # Initialize loss/accuracy variables
        losses = {'train': 0, 'val':0}
        accuracy = {'train': 0, 'val':0}
        counts = {'train': 0, 'val':0}

        # Process each split
        ii = 0
        val_ac = []
        for split in ['train', 'val']:
            # Set network mode
            if split == 'train':
                model.train()
            else:
                model.eval()

            # Process all split batches
            for i, (input, target, classes) in enumerate(loaders[split]):
                input = input.float()
                target = target.float()
                optimizer.zero_grad()
                with torch.set_grad_enabled(split == 'train'):
                    if torch.cuda.is_available():
                        output = model(input.to('cuda', non_blocking=True))
                    else:
                        output = model(input)
                    res = np.zeros((batch))
                    for i,val in enumerate(output):
                        t = val.cpu().detach().numpy()
                        if t[0]>t[1]:
                            res[i] = 0.
                        else:
                            res[i] = 1.
                    counts[split] += int(sum(res == classes.cpu().numpy()))
                    batch_total = res.size# labels.size(0) returns int
                    acc = round((counts[split] / batch_total) / 100, 2)
                    target = target.long()
                    output = output.double()
                    loss = criterion(output.to(device), target.to(device))
                    losses[split] += loss.item()
                    if split=='val':
                        val_ac.append(acc)
                    if(split=='train'):
                        loss = Variable(loss, requires_grad=True)
                        loss.backward()
        scheduler.step()
        counts[split] += 1
                    
        print("Epoch {}/{}: Train_Loss={} Val_Loss={} Val_accuracy={}".format(epoch, max_epoch, 
                                            round(losses['train']/counts['train'],4),round(losses['val']/counts['val'],4),
                                            round(sum(val_ac)/len(val_ac),4)))
    print("End")








list_nonbaseline = ['../datasets/physio_data_new.csv', '../datasets/kinect_data_new.csv', '../datasets/facereader_data_new.csv','../datasets/full_feature_dataset.csv']
dict_datasets = {"Physio" : 0, "Kinect" : 1, "Face" : 2, "All_features" : 3}

print('Run_1: Physio data...')
X_tr, X_test, y_tr, y_test, labels, feats = dataset_preprocess("Physio", False)
input_size = int(X_tr.shape[1])


train_nn_torch(X_tr, y_tr, X_test, y_test, input_size)