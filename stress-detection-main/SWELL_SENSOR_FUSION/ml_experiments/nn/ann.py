import math
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
from torch.utils.data import DataLoader
#from torchvision import transforms, datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from time import strftime, localtime
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, plot_confusion_matrix

from sklearn import metrics
import os




sample_weight=None
normalize=None
display_labels=None
include_values=True
xticks_rotation='horizontal'
values_format=None
ax=None


def visualize_comfusion_matrix(y_true, y_pred, class_names, print_name):
    # Plot non-normalized confusion matrix
    titles_options = [print_name + " Confusion matrix"]
    for title in titles_options:
        cm = metrics.confusion_matrix(y_true, y_pred)

        dis = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                        display_labels=display_labels)
        disp = dis.plot(include_values=include_values,
                            cmap=plt.cm.Blues, ax=ax, xticks_rotation=xticks_rotation,
                            values_format=values_format)
        disp.ax_.set_title(title)
        plt.savefig(RESULTS_FOLDER + "/" + print_name.replace(' ', '_') + ".png")
        plt.close()


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()




class TrajectoriesDataset:  
    # Constructor
    def __init__(self, dataset, labels):
        # Load data
        self.data = dataset
        self.label = labels
        #self.data = scio.loadmat(dataset)['coords'][0];
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
        self.layer_1 = nn.Linear(input_size, hidden_size*5)
        self.RELU1 = nn.ReLU()
        self.layer_1_BN = nn.BatchNorm1d(hidden_size*5)
        self.layer_2 = nn.Linear(hidden_size*5, hidden_size*3)
        self.RELU2 = nn.ReLU()
        self.layer_2_BN = nn.BatchNorm1d(hidden_size*3)
        self.output = nn.Linear(hidden_size*3, output_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_1_BN(x)
        x = self.RELU1(x)
        c = self.layer_2(x)
        c = self.layer_2_BN(c)
        c = self.RELU2(c)
        out = self.output(c)
        out = self.sigmoid(out)
        out = self.softmax(out)
        return out



def train_and_eval(model, model_name):
    ####################################################################################
    ## Training Section
    # Start training Session
    training = True
    if training:
        print("Start training process")
        tot_loss = 0
        max_epoch = 100
        dist = np.zeros((max_epoch,len(X_test)))
        max_acc = -1
        loss_tr_plot = []
        loss_ev_plot = []
        for epoch in range(1, max_epoch+1):
            train_correct = 0
            train_total = 0
            running_loss = 0.0

            # Initialize loss/accuracy variables
            losses = {'train': 0, 'val':0}
            counts = {'train': 0, 'val':0}

            # Process each split
            ii = 0

            # Process all split batches
            model.train()
            model.to(device)
            train_losses = 0
            for i, (input, target, classes) in enumerate(loaders['train']):
                input = input.float()
                target = target.float()
                optimizer.zero_grad()
                output = model(input.to(device))
                loss = criterion(output.to(device), target.to(device))
                loss.backward()
                optimizer.step()
                train_losses += loss.item()
                counts['train'] += 1

            scheduler.step()
            model.eval()
            test_losses = 0
            acc = 0
            for i, (input, target, classes) in enumerate(loaders['val']):
                input = input.float()
                target = target.float()
                outputs_test = model(input.to(device))
                loss_test = criterion(outputs_test, target.to(device))
                test_losses += loss_test.item()
                counts['val'] += 1
                output = outputs_test.detach().cpu().numpy()
                maxk = np.argmax(output, axis=1)
                target_  = target.numpy()
                maxt = np.argmax(target_, axis=1)
                if maxk[0] == maxt[0]:
                    acc += 1
            if acc > max_acc:
                max_acc = acc
                torch.save(model,RESULTS_FOLDER + "/" + model_name)
                print("Model saved!")
            print("Epoch {}/{}: Train_Loss={} Val_Loss={} Acc={}".format(epoch, max_epoch, round(train_losses/counts['train'],4),round(test_losses/counts['val'],4), round(100*(acc/counts['val']), 2)))
            loss_tr_plot.append(round(train_losses/counts['train'],4))
            loss_ev_plot.append(round(test_losses/counts['val'],4))
        print("End")

        plt.plot(loss_tr_plot, label='Training loss')
        plt.plot(loss_ev_plot, label='Validation loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training-Validation Loss plot")
        plt.legend(frameon=False)
        plt.savefig(RESULTS_FOLDER + "/" + "plot_losses.png")
        plt.close()


    model = Model(input_size, 256, 2)
    model = torch.load(RESULTS_FOLDER + "/" + model_name)
    model = model.to(device)
    torch.no_grad()
    model.eval()
    test_losses = 0
    pre = []
    gt = []

    for i, (input, target, classes) in enumerate(loaders['val']):
        input = input.float()
        target = target.float()
        outputs_test = model(input.to(device))
        output = outputs_test.detach().cpu().numpy()
        maxk = np.argmax(output, axis=1)
        target_  = target.numpy()
        maxt = np.argmax(target_, axis=1)
        gt.append(maxt[0])
        pre.append(maxk[0])

    cm = metrics.confusion_matrix(gt, pre)
    print('Sensor Decision Fusion Comfusion Matrix:')
    print_cm(cm,['Relax','Stress'])


    visualize_comfusion_matrix(gt, pre, ['Relax','Stress'], "Confusion Matrix")
    tn, fp, fn, tp = cm.ravel()

    print('\n')
    print("Sensor Fusion")
    print('F1 Score:', round(100*float((2*tp)/(2*tp+fn+fp)),6),'%')
    print('Precision:', round(100*float(tp/(tp+fp)),6),'%')
    print('Recall:', round(100*float(tp/(tp+fn)),6),'%')
    print('Accuracy:', round(100*float((tp+tn)/(tp+tn+fn+fp)),6),'%')


def preprocess(pp, fl = False):

    st_scaler = preprocessing.StandardScaler()
    df_time = pd.read_csv(pp)

    voc = {"Condition": {"N":0, "I":1, "T":1, "R":0}}

    df_t = df_time
    df_t.replace(voc, inplace=True)

    labels = df_t.Condition

    dd = df_t.loc[df_t["PP"] < 4]

    df = df_t.drop(df_t.columns[[0,1,2,3,4]],axis=1)

    df = df.fillna(df.mean())
    df = df.replace([-np.inf], 0.0)
    if fl:
        df = df.drop(['ln_vlf'], axis=1)
    feats = df.columns
    st_scaler.fit(df)
    data = st_scaler.transform(df)
    #data = df.values

    sep = -1*dd.shape[0]
    labels = labels.values

    X_tr, X_test, y_tr, y_test = train_test_split(data, labels, test_size = 0.2, random_state = 42)
    input_size = df.shape[1]

    return X_tr, X_test, y_tr, y_test, input_size



def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def dataset_preprocess(pathin, pp_out=1, baseline = False):
    print("Load and preprocess %s data.." %pathin)
    if not baseline:
        df_time = pd.read_csv(pathin)
        voc = {"Condition": {"N":0, "I":1, "T":1, "R":0}}
    else:
        df_time = pd.read_csv(pathin)
        voc = {"Condition": {"N":0, "S":1}}

    # print(df_time.columns)
    st_scaler = preprocessing.StandardScaler()
    df_t = df_time
    df_t.replace(voc, inplace=True)
    
    dd = df_t.loc[df_t["PP"] == pp_out]

    labels = df_t["Condition"]
    print(labels.value_counts())
    df = df_t.drop(df_t.columns[[0,1,2,3,4]],axis=1)

    if "physio" in pathin:
        df = df.drop(['ln_vlf'], axis=1)
    if baseline:
        df = df.drop(df.columns[int(-1+df.shape[1])],axis=1)

    df = df.fillna(df.mean())
    df = df.replace([-np.inf], 0.0)
    feats = df.columns

    st_scaler.fit(df)
    data = st_scaler.transform(df)
    labels = labels.values

    ind_d = dd.index.tolist()
    fd = ind_d[0]
    ld = ind_d[-1]

    X_test = data[fd:ld, :]
    y_test = labels[fd:ld]

    X_tr = np.concatenate((data[0:fd,:], data[ld:, :]), axis=0)
    y_tr = np.concatenate((labels[0:fd], labels[ld:]), axis=0)
    
    X_tr, y_tr = unison_shuffled_copies(X_tr, y_tr)
    X_test, y_test = unison_shuffled_copies(X_test, y_test)

    unique, counts = np.unique(y_tr, return_counts=True)
    print("Train set")
    print(dict(zip(unique, counts)))
    unique, counts = np.unique(y_test, return_counts=True)
    print("Test set")
    print(dict(zip(unique, counts)))

    input_size = df.shape[1]
    return X_tr, X_test, y_tr, y_test, input_size






################ ALL DATA 
batch = 64

X_tr, X_test, y_tr, y_test, input_size = preprocess('../datasets/full_feature_dataset.csv')
dataset  = {'train':TrajectoriesDataset(dataset=X_tr, labels=y_tr), 'val':TrajectoriesDataset(dataset=X_test, labels=y_test)}

loaders = {}
loaders["train"] = DataLoader(dataset["train"], batch_size = batch, drop_last = True, shuffle = True)
loaders["val"] = DataLoader(dataset["val"], batch_size = 1, drop_last = True, shuffle = True)

RESULTS_FOLDER = "models/demo_presentation/part_out"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

print("ALL DATA")
print("Model initialization...")
model_name = "model_ANN_all.pt"

model = Model(input_size, 256, 2)

torch.cuda.current_device()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', device)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.00035)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
criterion = nn.BCELoss()
criterion.to(device)

train_and_eval(model, model_name)
print("####"*10)










# ####################################################################################

# X_tr, X_test, y_tr, y_test, input_size = preprocess('../datasets/physio_data_new.csv', True)
# dataset  = {'train':TrajectoriesDataset(dataset=X_tr, labels=y_tr), 'val':TrajectoriesDataset(dataset=X_test, labels=y_test)}
# loaders = {}
# loaders["train"] = DataLoader(dataset["train"], batch_size = batch, drop_last = True, shuffle = True)
# loaders["val"] = DataLoader(dataset["val"], batch_size = 1, drop_last = True, shuffle = True)




# RESULTS_FOLDER = "models/demo_presentation/physio"
# os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ## Model Section
# model_name = "model_ANN_physio.pt"

# print("Model initialization...")
# model = Model(input_size, 256, 2)

# torch.cuda.current_device()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print('Using device:', device)
# model.to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr = 0.0004)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
# criterion = nn.MSELoss()
# criterion.to(device)
# train_and_eval(model, model_name)

# ####################################################################################




# batch = 64

# X_tr, X_test, y_tr, y_test, input_size = preprocess('../datasets/kinect_data_new.csv')
# dataset  = {'train':TrajectoriesDataset(dataset=X_tr, labels=y_tr), 'val':TrajectoriesDataset(dataset=X_test, labels=y_test)}

# loaders = {}
# loaders["train"] = DataLoader(dataset["train"], batch_size = batch, drop_last = True, shuffle = True)
# loaders["val"] = DataLoader(dataset["val"], batch_size = 1, drop_last = True, shuffle = True)




# RESULTS_FOLDER = "models/demo_presentation/kinect"
# os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ## Model Section
# model_name = "model_ANN_kinect.pt"

# print("Model initialization KINECT...")
# model = Model(input_size, 256, 2)

# torch.cuda.current_device()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print('Using device:', device)
# model.to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr = 0.0004)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
# criterion = nn.MSELoss()
# criterion.to(device)
# train_and_eval(model, model_name)

# print("####"*10)

# ####################################################################################



# batch = 64
# X_tr, X_test, y_tr, y_test, input_size = preprocess('../datasets/facereader_data_new.csv')

# dataset  = {'train':TrajectoriesDataset(dataset=X_tr, labels=y_tr), 'val':TrajectoriesDataset(dataset=X_test, labels=y_test)}

# loaders = {}
# loaders["train"] = DataLoader(dataset["train"], batch_size = batch, drop_last = True, shuffle = True)
# loaders["val"] = DataLoader(dataset["val"], batch_size = 1, drop_last = True, shuffle = True)


# RESULTS_FOLDER = "models/demo_presentation/face"
# os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ## Model Section
# model_name = "model_ANN_face.pt"

# print("Model initialization FACE...")
# model = Model(input_size, 256, 2)

# torch.cuda.current_device()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print('Using device:', device)
# model.to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr = 0.0004)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
# criterion = nn.MSELoss()
# criterion.to(device)
# train_and_eval(model, model_name)
