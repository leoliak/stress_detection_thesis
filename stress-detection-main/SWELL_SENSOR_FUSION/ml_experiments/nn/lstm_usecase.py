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

####################################################################################


## Model Section
print("Model initialization...")
model = Model_LSTM(input_size, 256, 2)

torch.cuda.current_device()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', device)
device = "cpu"
model.to(device)

#optimizer = torch.optim.SGD(model.parameters(), lr=1, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0004)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)

criterion = nn.MSELoss()
# criterion = nn.CrossEntropyLoss()
criterion.to(device)



#####################################################################################
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
        train_losses = 0
        for i, (input, target, classes) in enumerate(loaders['train']):
            input = input.float()
            target = target.float()
            optimizer.zero_grad()
            output = model(input)
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
            outputs_test = model(input)
            loss_test = criterion(outputs_test, target)
            test_losses += loss_test.item()
            counts['val'] += 1
            output = outputs_test.detach().numpy()
            maxk = np.argmax(output, axis=1)
            target_  = target.numpy()
            maxt = np.argmax(target_, axis=1)
            if maxk[0] == maxt[0]:
                acc += 1
        if acc > max_acc:
            max_acc = acc
            torch.save(model, "model_ANN_v3.pt")
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
    plt.savefig("plot_losses.png")
    plt.close()

model = Model(input_size, 256, 2)
model = torch.load("model_ANN_v2_SGD.pt")
torch.no_grad()
model.eval()
test_losses = 0
pre = []
gt = []
for i, (input, target, classes) in enumerate(loaders['val']):
    input = input.float()
    target = target.float()
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















# ###################################################################################3

#                     ##################################
#                     #   Keras Model Implementation   #
#                     ##################################


# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.callbacks import History
# from keras.optimizers import SGD
# from keras.layers import Embedding
# from keras.layers import LSTM, Flatten


# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# X_tr, X_test, y_tr, y_test = train_test_split(data, labels, test_size = 0.2, random_state = 42)

# hist = History()

# a = np.zeros((y_tr.shape[0],2))
# for i, val in enumerate(y_tr):
#    if val == 0:
#        a[i][0] = 1
#    else:
#        a[i][1] = 1
# y_tr = a


# model1 = Sequential()
# model1.add(Dense(256, input_dim=input_size, activation='relu'))
# model1.add(Dropout(0.3))
# model1.add(Dense(512, activation='relu'))
# model1.add(Dropout(0.5))
# model1.add(Dense(512, activation='relu'))
# model1.add(Dropout(0.3))
# model1.add(Dense(256, activation='relu'))
# model1.add(Dense(2, activation='sigmoid'))

# model1.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# model1.fit(X_tr, y_tr, epochs=150, batch_size=32, validation_split = .1, callbacks = [hist])

# a = np.zeros((y_test.shape[0],2))
# y_test_2 = y_test
# for i, val in enumerate(y_test):
#    if val == 0:
#        a[i][0] = 1
#    else:
#        a[i][1] = 1
# y_test = a

# a_ss, accuracy = model1.evaluate(X_test, y_test)
# print('Accuracy: %.2f' % (accuracy*100))

# predictions_class = model1.predict_classes(X_test)
# predictions_prob = model1.predict_proba(X_test)
# c_m_net = metrics.confusion_matrix(y_test_2, predictions_class)
# print('\n')
# print('NN Comfusion Matrix:')
# print_cm(c_m_net,['Relax','Stress'])



# X_tr, X_test, y_tr, y_test = train_test_split(data, labels, test_size = 0.2, random_state = 42)

# max_features = 1024

# model2 = Sequential()
# model2.add(Embedding(max_features, output_dim=256))
# model2.add(LSTM(512))
# model2.add(Dropout(0.5))
# model1.add(Dense(512, activation='relu'))
# model1.add(Dropout(0.3))
# model1.add(Dense(256, activation='relu'))
# model1.add(Dropout(0.5))
# model2.add(Dense(1, activation='sigmoid'))

# model2.compile(loss='binary_crossentropy',
#              optimizer='rmsprop',
#              metrics=['accuracy'])

# model2.fit(X_tr, y_tr, batch_size=5, epochs=150, validation_split = .1)
# a_ss, accuracy2 = model2.evaluate(X_test, y_test)
# print('Accuracy: %.2f' % (accuracy2*100))

# predictions_class2 = model2.predict_classes(X_test)
# predictions_prob2 = model2.predict_proba(X_test)
# c_m_net2 = metrics.confusion_matrix(y_test_2, predictions_class2)
# print('\n')
# print('NN Comfusion Matrix:')
# print_cm(c_m_net2,['Relax','Stress'])







# X_tr, X_test, y_tr, y_test = train_test_split(data, labels, test_size = 0.2, random_state = 42)

# trainset = np.reshape(X_tr, (X_tr.shape[0], 1, X_tr.shape[1]))
# testset = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# hist3 = History()

# model3 = Sequential()
# model3.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2, input_shape=((1, 170))))
# model3.add(Dense(512, activation = 'relu'))
# model3.add(Dropout(0.5))
# model3.add(Dense(128, activation = 'relu'))
# model3.add(Dropout(0.5))
# model3.add(Dense(1, activation='sigmoid'))
# model3.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# model3.fit(trainset, y_tr, epochs=1000, batch_size=32, validation_split = .1, callbacks = [hist3], verbose=1,)
# a = np.zeros((y_test.shape[0],2))
# y_test_2 = y_test
# for i, val in enumerate(y_test):
#     if val == 0:
#         a[i][0] = 1
#     else:
#         a[i][1] = 1
# y_test = a

# a_ss, accuracy = model3.evaluate(testset, y_test_2, batch_size=5)
# print('Accuracy: %.2f' % (accuracy*100))


# # serialize model to JSON
# # model_json = model3.to_json()
# # with open("nn_trial\model_3.json", "w") as json_file:
# #     json_file.write(model_json)
# # # serialize weights to HDF5
# # model3.save_weights("nn_trial\model_3.h5")
# # print("Saved model to disk")
 
# # 
# ## load json and create model
# #from keras.models import model_from_json
# #json_file = open('model.json', 'r')
# #loaded_model_json = json_file.read()
# #json_file.close()
# #loaded_model = model_from_json(loaded_model_json)
# ## load weights into new model
# #loaded_model.load_weights("model.h5")
# #print("Loaded model from disk")
