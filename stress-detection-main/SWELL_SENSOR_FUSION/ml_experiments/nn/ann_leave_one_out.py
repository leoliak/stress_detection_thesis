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
    
    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_1_BN(x)
        x = self.RELU1(x)
        c = self.layer_2(x)
        c = self.layer_2_BN(c)
        c = self.RELU2(c)
        out = self.output(c)
        out = self.sigmoid(out)
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
            model.to(device)
            model.train()
            train_losses = 0
            for i, (input, target, _) in enumerate(loaders['train']):
                input = input.float()
                target = target.float()
                target = target.to(device)
                optimizer.zero_grad()
                output = model(input.to(device))
                loss = criterion(output, target)
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
                target = target.to(device)
                input = input.to(device)
                outputs_test = model(input)
                loss_test = criterion(outputs_test, target)
                test_losses += loss_test.item()
                counts['val'] += 1
                output = outputs_test.cpu().detach().numpy()
                maxk = np.argmax(output, axis=1)
                target_  = target.cpu().numpy()
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
    print(RESULTS_FOLDER + "/" + model_name)
    model = torch.load(RESULTS_FOLDER + "/" + model_name)
    torch.no_grad()
    model.eval()
    test_losses = 0
    pre = []
    gt = []
    model.cpu()
    for i, (input, target, _) in enumerate(loaders['val']):
        input = input.float()
        target = target.float()
        target = target.cpu()
        outputs_test = model(input)
        output = outputs_test.detach().numpy()
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

    f1s = round(100*float((2*tp)/(2*tp+fn+fp)),6)
    rec = round(100*float(tp/(tp+fp)),6)
    pre = round(100*float(tp/(tp+fn)),6)
    acc = round(100*float((tp+tn)/(tp+tn+fn+fp)),6)

    return f1s, rec, pre, acc, model




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




def dataset_preprocess_fusion_2(pathin, pp_out = -1):
    datas_i = []
    label_i = []
    di = {}
    baseline = False
    if not baseline:
        df_time = pd.read_csv(pathin)
        voc = {"Condition": {"N":0, "I":1, "T":1, "R":0}}

    # print(df_time.columns)
    st_scaler = preprocessing.StandardScaler()
    df_t = df_time
    df_t.replace(voc, inplace=True)
    labels = df_t["Condition"]
    print(labels.value_counts())
    df = df_t.drop(df_t.columns[[0,1,2,3,4]],axis=1)
    df = df.drop(['ln_vlf'], axis=1)
    # df = df.drop(['SD1','SD2','SDSD'],axis=1)
    df = df.fillna(df.mean())
    df = df.replace([-np.inf], 0.0)

    dd = df_t.loc[df_t["PP"] == pp_out]
    ind_d = dd.index.tolist()
    fd = ind_d[0]
    ld = ind_d[-1]

    st_scaler.fit(df)
    data = st_scaler.transform(df)
    lim_1 = 34
    lim_2 = 128
    X_physio_ = data[:, 0:lim_1]
    X_kinect_ = data[:, lim_1:lim_2]
    X_face_ = data[:, lim_2:]
    y_labels = labels.values
    
    X_physio = X_physio_[fd:ld, :]
    X_kinect = X_kinect_[fd:ld, :]
    X_face = X_face_[fd:ld, :]
    y_test = labels[fd:ld]
    return X_physio, X_kinect, X_face, y_test



def list2int(dlist):
    olist = [int(x) for x in dlist]
    return(olist)



# Physio, Kinect, Face
def sensor_fusion_decision(physio, kinect, face, strategy):
    decision_fusion = []
    labels = []
    physio_, kinect_, face_ = list2int(physio[0,:].tolist()), list2int(kinect[0,:].tolist()), list2int(face[0,:].tolist())
    if strategy == "Majority":
        for i, (p, k, f) in enumerate(zip(physio_, kinect_, face_)):
            if p+k+f <= 1:
                decision = 0
            else:
                decision = 1
            decision_fusion.append(decision)
            labels.append([p, k, f])
        return decision_fusion, labels
    elif strategy == "weighted-majority":
        for i, (p, k, f) in enumerate(zip(physio_, kinect_, face_)):
            if p == 0:
                p = -1*physio[1,i]
            else:
                p = 1*physio[1,i]
            if k == 0:
                k = -1*kinect[1,i]
            else:
                k = 1*kinect[1,i]
            if f == 0:
                f = -1*face[1,i]
            else:
                f = 1*face[1,i]
            decision = p + k + f
            if decision<=0:
                decision_fusion.append(0)
            else:
                decision_fusion.append(1)
            labels.append([p, k, f])
        return decision_fusion, labels
    else:
        print("ERROR picking strategy")
        return -1

def cross_mulitclass(clf, X, y, K):
    from sklearn.model_selection import cross_validate
    from sklearn.metrics import recall_score
    scoring = ['precision', 'recall', 'accuracy', 'f1']
    scores = cross_validate(clf, X, y, cv=K, scoring=scoring)
    keys = list(scores.keys())
    print("Results from CV")
    for key in keys:
        data = scores[key]
        print(key + ": {}".format(100*round(sum(data)/len(data), 4)))
    print('\n')
    return scores




def pred_NN(data, model):
    zz = np.zeros((2, data.shape[0]))
    torch.no_grad()
    model.eval()
    pre = []
    pre1 = []
    pre2 = []
    m = nn.Softmax(dim=1)
    for i, input in enumerate(data):
        input = torch.from_numpy(input)
        input = input.float()
        input = input.unsqueeze(dim=0)
        outputs_test = model(input)
        prob = m(outputs_test)
        outputs_test = outputs_test.detach().cpu().numpy()
        prob = prob.detach().cpu().numpy().squeeze(0)
        maxk = np.argmax(outputs_test, axis=1)
        pre.append([maxk[0], prob[maxk[0]]])   
        zz[0, i] = maxk[0]
        zz[1, i] = prob[maxk[0]]
    return zz










################ ALL DATA ########


batch = 64
f1_score_all = []
recall_all = []
precision_all = []
accuracy_all = []

di_majority_f1 = []
di_majority_rec = []
di_majority_pre = []
di_majority_acc = []

di_weighted_f1 = []
di_weighted_rec = []
di_weighted_pre = []
di_weighted_acc = []

f1_score_physio = []
recall_physio = []
precision_physio = []
accuracy_physio = []

f1_score_kinect = []
recall_kinect = []
precision_kinect = []
accuracy_kinect = []

f1_score_face = []
recall_face = []
precision_face = []
accuracy_face = []



fusion_FL = True
keys = [2,3,4,5,6,7,9,10,12,13,14,15,16,17,18,19,20,21,22,24,25]
for key in keys:
    pathin = 'datasets/full_feature_dataset.csv'
    X_tr, X_test, y_tr, y_test, input_size = dataset_preprocess(pathin, pp_out=key)
    dataset  = {'train':TrajectoriesDataset(dataset=X_tr, labels=y_tr), 'val':TrajectoriesDataset(dataset=X_test, labels=y_test)}

    loaders = {}
    loaders["train"] = DataLoader(dataset["train"], batch_size = batch, drop_last = True, shuffle = True)
    loaders["val"] = DataLoader(dataset["val"], batch_size = 1, drop_last = True, shuffle = True)

    RESULTS_FOLDER = "ANN/models/demo_presentation/part_out"
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    print("ALL DATA")
    print("Model initialization...")
    model_name = "model_ANN_all.pt"

    model_all = Model(input_size, 256, 2)

    torch.cuda.current_device()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    model_all.to(device)

    optimizer = torch.optim.Adam(model_all.parameters(), lr = 0.00035)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    criterion = nn.MSELoss()
    criterion.to(device)

    f1s, rec, pre, acc, model_all = train_and_eval(model_all, model_name)
    f1_score_all.append(f1s)
    recall_all.append(rec)
    precision_all.append(pre)
    accuracy_all.append(acc)
    print("####"*10)



    ####################################################################################

    pathin = 'datasets/physio_data_new.csv'
    X_tr, X_test, y_tr, y_test, input_size = dataset_preprocess(pathin, pp_out=key)
    # X_tr, X_test, y_tr, y_test, input_size = preprocess('../datasets/physio_data_new.csv', True)
    dataset  = {'train':TrajectoriesDataset(dataset=X_tr, labels=y_tr), 'val':TrajectoriesDataset(dataset=X_test, labels=y_test)}
    loaders = {}
    loaders["train"] = DataLoader(dataset["train"], batch_size = batch, drop_last = True, shuffle = True)
    loaders["val"] = DataLoader(dataset["val"], batch_size = 1, drop_last = True, shuffle = True)

    RESULTS_FOLDER = "ANN/models/part_out/physio"
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    ## Model Section
    model_name = "model_ANN_physio.pt"

    print("Model initialization...")
    model_physio = Model(input_size, 256, 2)

    torch.cuda.current_device()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    model_physio.to(device)

    optimizer = torch.optim.Adam(model_physio.parameters(), lr = 0.0004)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
    criterion = nn.MSELoss()
    criterion.to(device)
    f1s, rec, pre, acc, model_physio = train_and_eval(model_physio, model_name)
    f1_score_physio.append(f1s)
    recall_physio.append(rec)
    precision_physio.append(pre)
    accuracy_physio.append(acc)

    ####################################################################################

    pathin = 'datasets/kinect_data_new.csv'
    X_tr, X_test, y_tr, y_test, input_size = dataset_preprocess(pathin, pp_out=key)
    # X_tr, X_test, y_tr, y_test, input_size = preprocess('../datasets/kinect_data_new.csv')
    dataset  = {'train':TrajectoriesDataset(dataset=X_tr, labels=y_tr), 'val':TrajectoriesDataset(dataset=X_test, labels=y_test)}

    loaders = {}
    loaders["train"] = DataLoader(dataset["train"], batch_size = batch, drop_last = True, shuffle = True)
    loaders["val"] = DataLoader(dataset["val"], batch_size = 1, drop_last = True, shuffle = True)

    RESULTS_FOLDER = "ANN/models/part_out/kinect"
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    ## Model Section
    model_name = "model_ANN_kinect.pt"

    print("Model initialization KINECT...")
    model_kinect = Model(input_size, 256, 2)

    torch.cuda.current_device()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    model_kinect.to(device)

    optimizer = torch.optim.Adam(model_kinect.parameters(), lr = 0.0004)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
    criterion = nn.MSELoss()
    criterion.to(device)
    f1s, rec, pre, acc, model_kinect = train_and_eval(model_kinect, model_name)
    f1_score_kinect.append(f1s)
    recall_kinect.append(rec)
    precision_kinect.append(pre)
    accuracy_kinect.append(acc)

    print("####"*10)

    ####################################################################################

    pathin = 'datasets/facereader_data_new.csv'
    X_tr, X_test, y_tr, y_test, input_size = dataset_preprocess(pathin, pp_out=key)  
    dataset  = {'train':TrajectoriesDataset(dataset=X_tr, labels=y_tr), 'val':TrajectoriesDataset(dataset=X_test, labels=y_test)}

    loaders = {}
    loaders["train"] = DataLoader(dataset["train"], batch_size = batch, drop_last = True, shuffle = True)
    loaders["val"] = DataLoader(dataset["val"], batch_size = 1, drop_last = True, shuffle = True)


    RESULTS_FOLDER = "ANN/models/part_out/face"
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    ## Model Section
    model_name = "model_ANN_face.pt"

    print("Model initialization FACE...")
    model_face = Model(input_size, 256, 2)

    torch.cuda.current_device()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    model_face.to(device)

    optimizer = torch.optim.Adam(model_face.parameters(), lr = 0.0004)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
    criterion = nn.MSELoss()
    criterion.to(device)
    f1s, rec, pre, acc, model_face = train_and_eval(model_face, model_name)
    f1_score_face.append(f1s)
    recall_face.append(rec)
    precision_face.append(pre)
    accuracy_face.append(acc)

    ########################################################################

    if fusion_FL:
        pathin = 'datasets/full_feature_dataset.csv'
        dataset  = {'train':TrajectoriesDataset(dataset=X_tr, labels=y_tr), 'val':TrajectoriesDataset(dataset=X_test, labels=y_test)}

        X_physio, X_kinect, X_face, y_labels  = dataset_preprocess_fusion_2(pathin, pp_out=key)

        d1_NN = pred_NN(X_physio, model_physio)
        d2_NN = pred_NN(X_kinect, model_kinect)
        d3_NN = pred_NN(X_face, model_face)

        #########################################################
        ######### MAJORITY FUSION SENSOR ARCHITECTUERS ##########
        #########################################################

        #####
        print("Sensor Fusion - All NNs")
        decision_system_classification, mds_labels = \
                            sensor_fusion_decision(d1_NN, d2_NN, d3_NN, "Majority")
        dsf = np.array(decision_system_classification)
        print(metrics.classification_report(y_labels, dsf))


        c_m_dsf = metrics.confusion_matrix(y_labels, dsf)
        print('Sensor Decision Fusion Comfusion Matrix:')
        print_cm(c_m_dsf,['Relax','Stress'])
            
        tn, fp, fn, tp = c_m_dsf.ravel()
        print('\n')
        print('F1 Score:', round(100*float((2*tp)/(2*tp+fn+fp)),6),'%')
        print('Precision:', round(100*float(tp/(tp+fp)),6),'%')
        print('Recall:', round(100*float(tp/(tp+fn)),6),'%')
        print('Accuracy:', round(100*float((tp+tn)/(tp+tn+fn+fp)),6),'%')

        di_majority_f1.append(round(100*float((2*tp)/(2*tp+fn+fp)),6))
        di_majority_rec.append( round(100*float(tp/(tp+fp)),6))
        di_majority_pre.append(round(100*float(tp/(tp+fn)),6))
        di_majority_acc.append(round(100*float((tp+tn)/(tp+tn+fn+fp)),6))


        #########################################################
        ######### WEIGHTED FUSION SENSOR ARCHITECTUERS ##########
        #########################################################


        #####
        print("Sensor Fusion - Weighted Majority - All NN")

        decision_system_classification, mds_labels = \
            sensor_fusion_decision(d1_NN, d2_NN, d3_NN, "weighted-majority")
        dsf = np.array(decision_system_classification)
        print(metrics.classification_report(y_labels, dsf))

        c_m_dsf = metrics.confusion_matrix(y_labels, dsf)
        print('Sensor Decision Fusion Comfusion Matrix Weighted Majority:')
        print_cm(c_m_dsf,['Relax','Stress'])
            
        tn, fp, fn, tp = c_m_dsf.ravel()
        print('\n')
        print('F1 Score:', round(100*float((2*tp)/(2*tp+fn+fp)),6),'%')
        print('Precision:', round(100*float(tp/(tp+fp)),6),'%')
        print('Recall:', round(100*float(tp/(tp+fn)),6),'%')
        print('Accuracy:', round(100*float((tp+tn)/(tp+tn+fn+fp)),6),'%')

        di_weighted_f1.append(round(100*float((2*tp)/(2*tp+fn+fp)),6))
        di_weighted_rec.append( round(100*float(tp/(tp+fp)),6))
        di_weighted_pre.append(round(100*float(tp/(tp+fn)),6))
        di_weighted_acc.append(round(100*float((tp+tn)/(tp+tn+fn+fp)),6))



##########################################################################
##########################################################################

print("####"*20)
print("RUN 1: All Features")
print("Total F1-score: {}".format(sum(f1_score_all)/len(f1_score_all)))
print("Total Recall: {}".format(sum(recall_all)/len(recall_all)))
print("Total Precision: {}".format(sum(precision_all)/len(precision_all)))
print("Total Accuracy: {}".format(sum(accuracy_all)/len(accuracy_all)))

print("RUN 2: Physio Features")
print("Total F1-score: {}".format(sum(f1_score_physio)/len(f1_score_physio)))
print("Total Recall: {}".format(sum(recall_physio)/len(recall_physio)))
print("Total Precision: {}".format(sum(precision_physio)/len(precision_physio)))
print("Total Accuracy: {}".format(sum(accuracy_physio)/len(accuracy_physio)))

print("RUN 3: Kinect Features")
print("Total F1-score: {}".format(sum(f1_score_kinect)/len(f1_score_kinect)))
print("Total Recall: {}".format(sum(recall_kinect)/len(recall_kinect)))
print("Total Precision: {}".format(sum(precision_kinect)/len(precision_kinect)))
print("Total Accuracy: {}".format(sum(accuracy_kinect)/len(accuracy_kinect)))

print("RUN 4: Face Features")
print("Total F1-score: {}".format(sum(f1_score_face)/len(f1_score_face)))
print("Total Recall: {}".format(sum(recall_face)/len(recall_face)))
print("Total Precision: {}".format(sum(precision_face)/len(precision_face)))
print("Total Accuracy: {}".format(sum(accuracy_face)/len(accuracy_face)))

print("RUN 5: Decision Fusion - Majority")
print("Total F1-score: {}".format(sum(di_majority_f1)/len(di_majority_f1)))
print("Total Recall: {}".format(sum(di_majority_rec)/len(di_majority_rec)))
print("Total Precision: {}".format(sum(di_majority_pre)/len(di_majority_pre)))
print("Total Accuracy: {}".format(sum(di_majority_acc)/len(di_majority_acc)))

print("RUN 6: Decision Fusion - Weighted Majority")
print("Total F1-score: {}".format(sum(di_weighted_f1)/len(di_weighted_f1)))
print("Total Recall: {}".format(sum(di_weighted_rec)/len(di_weighted_rec)))
print("Total Precision: {}".format(sum(di_weighted_pre)/len(di_weighted_pre)))
print("Total Accuracy: {}".format(sum(di_weighted_acc)/len(di_weighted_acc)))