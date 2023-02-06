# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 20:30:23 2019

@author: leoni
"""

"""
Dokimes panw sta time data poy ekana extraction. H klasi Classifiers epistrefei ta apotelesmata
apo hyperparameter tuning gia 5 algorithmous mixanikis mathisis. Epistrefei to modelo, ton meso oro tou
10 fold validation test kai tis parametrous poy epilexthikan ws oi kaliteres apo ton algorithmo.

Sto telos kanw test gia tous 3 kalyterous me to test set poy exw, to train set kathws kai to sinolo toy dataset
"""
from time import strftime, localtime
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.svm import SVC,LinearSVC
from sklearn.linear_model import LogisticRegression as lr
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split,cross_val_score,KFold,StratifiedKFold
from sklearn import metrics
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2, RFE
import pickle
import sys


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
        
        
        
class Classifiers(object):
    def __init__(self,train_data,train_labels,use_lin,hyperTune=True):
        self.train_data=train_data
        self.train_labels=train_labels
        self.use_lin = use_lin
        self.construct_all_models(hyperTune,use_lin)
    
    def construct_all_models(self,hyperTune,use_lin):
        if hyperTune:
            #3 models KNN SCM and LR
            if use_lin:
                self.models={'SVM':[SVC(probability=True),dict(kernel=['rbf'],gamma=np.logspace(-3, 1, 5),C=np.arange(0.01, 3.01, 0.1))],\
                             'LinearSVM':[LinearSVC(dual=False),dict(penalty=['l1','l2'],tol=np.logspace(-8, -2, 7),C=np.arange(0.01, 3.01, 0.2))],\
                             'LogisticRegression':[lr(solver='lbfgs',multi_class='auto'),dict(C=np.arange(0.1,3,0.1))],\
                             'KNN':[KNeighborsClassifier(),dict(n_neighbors=np.arange(1, 100))],\
                             'Random_Forest':[RandomForestClassifier(),dict(n_estimators = np.arange(200,1300,100),max_depth = np.linspace(20, 50, 7, endpoint=True))]}
            else:
              self.models={'SVM':[SVC(kernel='rbf',probability=True),dict(gamma=np.logspace(-9, 1, 11),C=np.arange(0.01, 3.01, 0.2))],\
                     'LogisticRegression':[lr(solver='lbfgs',multi_class='auto'),dict(C=np.arange(0.1,3,0.1))],\
                     'KNN':[KNeighborsClassifier(),dict(n_neighbors=np.arange(1, 100))],\
                     'Random_Forest':[RandomForestClassifier(),dict(n_estimators = np.arange(200,1300,100), max_depth = np.linspace(20, 50, 10, endpoint=True))]}
            for name,candidate_hyperParam in self.models.items():
                #update each classifier after training and tuning
                self.models[name] = self.train_with_hyperParamTuning(candidate_hyperParam[0],name,candidate_hyperParam[1])
            print ('\nTraining process finished\n')
            
    def train_with_hyperParamTuning(self,model,name,param_grid):
        #grid search method for hyper-parameter tuning
        grid = GridSearchCV(model, param_grid, cv=10, scoring='accuracy', n_jobs=4)
        grid.fit(self.train_data, self.train_labels)
        print(
            '\nThe best hyper-parameter for  {} is {}, mean accuracy through 10 Fold test is {} \n'\
            .format(name, grid.best_params_, round(100*grid.best_score_,2)))

        model = grid.best_estimator_
        score = grid.best_score_
        parameters = grid.best_params_
        train_pred = model.predict(self.train_data)
        print('{} train accuracy = {}\n'.format(name,100*(train_pred == self.train_labels).mean()))
        return model,score,parameters

    def prediction_metrics(self,test_data,test_labels,name):
        #accuracy
        print('{} test accuracy = {}\n'.format(name,100*(self.models[name].predict(test_data) == test_labels).mean()))
        #AUC of ROC
#        prob = self.models[name].predict_proba(test_data)
#        auc=roc_auc_score(test_labels.reshape(-1),prob[:,1])
#        auc=roc_auc_score(test_labels,prob[:,1])
#        print('Classifier {} area under curve of ROC is {}\n'.format(name,100*auc))
#        #ROC
#        fpr, tpr, thresholds = roc_curve(test_labels.reshape(-1), prob[:,1], pos_label=1)
#        self.roc_plot(fpr,tpr,name,auc)
#    
#    def roc_plot(self,fpr,tpr,name,auc):
#        plt.figure(figsize=(20,5))
#        plt.plot(fpr,tpr)
#        plt.ylim([0.0,1.0])
#        plt.ylim([0.0, 1.0])
#        plt.title('ROC of {}     AUC: {}\nPlease close it to continue'.format(name,auc))
#        plt.xlabel('False Positive Rate')
#        plt.ylabel('True Positive Rate')
#        plt.grid(True)
#        plt.show()



overall_dict= {}
comfusion_matrix = []
test_acc = []
fold_10_acc = []
model_par = []

print('--------------------------------------------------------------------')
## Keep time of running code
tim = strftime("%Y-%m-%d %H:%M:%S", localtime())
print(tim)


print('Run_3: All data included, and EDA...')
st_scaler = preprocessing.StandardScaler()

df_time = pd.read_csv('D:\Codes\\data\\full_feature_baseline_dataset.csv')

#df2 = pd.read_csv('time_feat_data.csv')
#df2 = df2.loc[:, ~df2.columns.str.match('Unnamed')]

voc = {"Condition": {"N":1, "S":2}}
df_t = df_time
df_t.replace(voc, inplace=True)

df_skin = df_t[['Skin_mean','Skin_dev','Skin_var','Skin_max','Skin_min']]
labels = df_t.Condition
df = df_t.drop(df_t.columns[[0,1,2,3,4]],axis=1)
df = df.drop(['ln_vlf'],axis=1)
#df = df.drop(['SD1','SD2','SDSD'],axis=1)
#df = df.drop(['Skin_mean','Skin_dev','Skin_var','Skin_max','Skin_min'],axis=1)

df = df.fillna(df.mean())
df = df.replace([-np.inf], 0.0)

feats = df.columns
st_scaler.fit(df)
data = st_scaler.transform(df)

X_tr, X_test, y_tr, y_test = train_test_split(data, labels, test_size = 0.3, random_state = 42)

print('Start training phase..')
classifiers = Classifiers(X_tr, y_tr,True)
models = classifiers.models
overall_dict['all_data'] = models
## Save model, mean score and parameters
save = False
if save==True:
    pkl_filename = "saved_models\model_time_data.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(models, file)
        print('Model Dictionary saved')
###   SVM train,test and whole dataset predict accuracy metrics
print('SVM results...')
svm_dict = models['SVM']
svm = svm_dict[0]
la = svm.predict(X_test)
la2 = svm.predict(X_tr)
la3 = svm.predict(data)

print('10 fold test:', 100*svm_dict[1])
print('Test accuracy:',100*(la==y_test).mean())
print('Train accuracy:',100*(la2==y_tr).mean())
print('Dataset accuracy:',100*(la3==labels).mean())

###   Random Forrest train,test and whole dataset predict accuracy metrics
rf_dict = models['Random_Forest']
print('Random Forrest results...')
rf = rf_dict[0]
la = rf.predict(X_test)
la2 = rf.predict(X_tr)
la3 = rf.predict(data)
print('10 fold test:', 100*rf_dict[1])
print('Test accuracy:',100*(la==y_test).mean())
print('Train accuracy:',100*(la2==y_tr).mean())
print('Dataset accuracy:',100*(la3==labels).mean())

###   KNN train,test and whole dataset predict accuracy metrics
knn_dict = models['KNN']
print('KNN results...')
knn = knn_dict[0]
la = knn.predict(X_test)
la2 = knn.predict(X_tr)
la3 = knn.predict(data)
print('10 fold test:', 100*knn_dict[1])
print('Test accuracy:',100*(la==y_test).mean())
print('Train accuracy:',100*(la2==y_tr).mean())
print('Dataset accuracy:',100*(la3==labels).mean())
print('\n\n---------------------------------------------------------------')


y_pred_svm = svm.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_knn = knn.predict(X_test)
test_acc.append([y_pred_svm,y_pred_rf,y_pred_knn])

a_svm = metrics.accuracy_score(y_test, y_pred_svm)
a_rf = metrics.accuracy_score(y_test, y_pred_rf)
a_knn = metrics.accuracy_score(y_test, y_pred_knn)

print('\n')
print('Test Score for SVM:', a_svm)
print('Test Score for RF:',a_rf)
print('Test Score for KNN:',a_knn)
c_m_svm = metrics.confusion_matrix(y_test, y_pred_svm)
c_m_rf = metrics.confusion_matrix(y_test, y_pred_rf)
c_m_knn = metrics.confusion_matrix(y_test, y_pred_knn)
print('\n')
print('SVM Comfusion Matrix:')
print_cm(c_m_svm,['Relax','Stress'])
print('\n')
print('RF Comfusion Matrix:')
print_cm(c_m_rf,['Relax','Stress'])
print('\n')
print('KNN Comfusion Matrix:')
print_cm(c_m_knn,['Relax','Stress'])
comfusion_matrix.append('SVM all datasets')
comfusion_matrix.append(c_m_svm)
comfusion_matrix.append('RF all datasets')
comfusion_matrix.append(c_m_rf)
comfusion_matrix.append('KNN all datasets')
comfusion_matrix.append(c_m_knn)


feat_importance = rf.feature_importances_
dictionary = dict(zip(feats, feat_importance))
x = dictionary
sorted_x = sorted(x.items(), key=lambda kv: kv[1],reverse=True)

print('\n\n\n\n---------------------------------------------------------------')
print('No Kinect data now...')

## Uncomment here for merge time and frequency features
st_scaler = preprocessing.StandardScaler()

df_time = pd.read_csv('full_dataset.csv')

voc = {"Condition": {"N":1, "I":2, "T":2, "R":1}}
df_t = df_time
df_t.replace(voc, inplace=True)

df_skin = df_t[['Skin_mean','Skin_dev','Skin_var','Skin_max','Skin_min']]
labels = df_t.Condition
df = df_t.drop(df_t.columns[[0,1,2,3]],axis=1)
#df = df.drop(['SD1','SD2','SDSD'],axis=1)
#df = df.drop(['Skin_mean','Skin_dev','Skin_var','Skin_max','Skin_min'],axis=1)

df = df.fillna(df.mean())
df = df.replace([-np.inf], 0.0)

feats = df.columns
st_scaler.fit(df)
data = st_scaler.transform(df)

X_tr, X_test, y_tr, y_test = train_test_split(data, labels, test_size = 0.3, random_state = 42)

print('Start training phase..')
classifiers = Classifiers(X_tr, y_tr,True)
models = classifiers.models
overall_dict['physio_n_kinect_data'] = models
## Save model, mean score and parameters
save = False
if save==True:
    pkl_filename = "saved_models\model_time_data.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(models, file)
        print('Model Dictionary saved')
###   SVM train,test and whole dataset predict accuracy metrics
print('SVM results...')
svm_dict = models['SVM']
svm = svm_dict[0]
la = svm.predict(X_test)
la2 = svm.predict(X_tr)
la3 = svm.predict(data)

print('10 fold test:', 100*svm_dict[1])
print('Test accuracy:', 100*(la==y_test).mean())
print('Train accuracy:', 100*(la2==y_tr).mean())
print('Dataset accuracy:', 100*(la3==labels).mean())

###   Random Forrest train,test and whole dataset predict accuracy metrics
rf_dict = models['Random_Forest']
print('Random Forrest results...')
rf = rf_dict[0]
la = rf.predict(X_test)
la2 = rf.predict(X_tr)
la3 = rf.predict(data)
print('10 fold test:', 100*rf_dict[1])
print('Test accuracy:',100*(la==y_test).mean())
print('Train accuracy:',100*(la2==y_tr).mean())
print('Dataset accuracy:',100*(la3==labels).mean())

###   KNN train,test and whole dataset predict accuracy metrics
knn_dict = models['KNN']
print('KNN results...')
knn = knn_dict[0]
la = knn.predict(X_test)
la2 = knn.predict(X_tr)
la3 = knn.predict(data)
print('10 fold test:', 100*knn_dict[1])
print('Test accuracy:',100*(la==y_test).mean())
print('Train accuracy:',100*(la2==y_tr).mean())
print('Dataset accuracy:',100*(la3==labels).mean())
print('\n\n---------------------------------------------------------------')


y_pred_svm = svm.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_knn = knn.predict(X_test)
test_acc.append([y_pred_svm,y_pred_rf,y_pred_knn])

a_svm = metrics.accuracy_score(y_test, y_pred_svm)
a_rf = metrics.accuracy_score(y_test, y_pred_rf)
a_knn = metrics.accuracy_score(y_test, y_pred_knn)

print('\n')
print('Test Score for SVM:', a_svm)
print('Test Score for RF:',a_rf)
print('Test Score for KNN:',a_knn)
c_m_svm = metrics.confusion_matrix(y_test, y_pred_svm)
c_m_rf = metrics.confusion_matrix(y_test, y_pred_rf)
c_m_knn = metrics.confusion_matrix(y_test, y_pred_knn)
print('\n')
print('SVM Comfusion Matrix:')
print_cm(c_m_svm,['Relax','Stress'])
print('\n')
print('RF Comfusion Matrix:')
print_cm(c_m_rf,['Relax','Stress'])
print('\n')
print('KNN Comfusion Matrix:')
print_cm(c_m_knn,['Relax','Stress'])

comfusion_matrix.append('SVM no kin dataset')
comfusion_matrix.append(c_m_svm)
comfusion_matrix.append('RF no kin dataset')
comfusion_matrix.append(c_m_rf)
comfusion_matrix.append('KNN no kin dataset')
comfusion_matrix.append(c_m_knn)




print('\n\n\n\n---------------------------------------------------------------')
print('Only physion data now...')

## Uncomment here for merge time and frequency features
st_scaler = preprocessing.StandardScaler()

df_time = pd.read_csv('physio_dataset.csv')
df_time = df_time.drop(['Condition.1','timestamp.1'],axis = 1)
voc = {"Condition": {"N":1, "I":2, "T":2, "R":1}}
df_t = df_time
df_t.replace(voc, inplace=True)

df_skin = df_t[['Skin_mean','Skin_dev','Skin_var','Skin_max','Skin_min']]
labels = df_t.Condition
df = df_t.drop(df_t.columns[[0,1,2,3]],axis=1)
#df = df.drop(['SD1','SD2','SDSD'],axis=1)
#df = df.drop(['Skin_mean','Skin_dev','Skin_var','Skin_max','Skin_min'],axis=1)

df = df.fillna(df.mean())
df = df.replace([-np.inf], 0.0)

feats = df.columns
st_scaler.fit(df)
data = st_scaler.transform(df)

X_tr, X_test, y_tr, y_test = train_test_split(data, labels, test_size = 0.3, random_state = 42)

print('Start training phase..')
classifiers = Classifiers(X_tr, y_tr,True)
models = classifiers.models
overall_dict['physio_data'] = models

## Save model, mean score and parameters
save = False
if save==True:
    pkl_filename = "saved_models\model_time_data.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(models, file)
        print('Model Dictionary saved')
###   SVM train,test and whole dataset predict accuracy metrics
print('SVM results...')
svm_dict = models['SVM']
svm = svm_dict[0]
la = svm.predict(X_test)
la2 = svm.predict(X_tr)
la3 = svm.predict(data)

print('10 fold test:', 100*svm_dict[1])
print('Test accuracy:', 100*(la==y_test).mean())
print('Train accuracy:', 100*(la2==y_tr).mean())
print('Dataset accuracy:', 100*(la3==labels).mean())

###   Random Forrest train,test and whole dataset predict accuracy metrics
rf_dict = models['Random_Forest']
print('Random Forrest results...')
rf = rf_dict[0]
la = rf.predict(X_test)
la2 = rf.predict(X_tr)
la3 = rf.predict(data)
print('10 fold test:', 100*rf_dict[1])
print('Test accuracy:',100*(la==y_test).mean())
print('Train accuracy:',100*(la2==y_tr).mean())
print('Dataset accuracy:',100*(la3==labels).mean())

###   KNN train,test and whole dataset predict accuracy metrics
knn_dict = models['KNN']
print('KNN results...')
knn = knn_dict[0]
la = knn.predict(X_test)
la2 = knn.predict(X_tr)
la3 = knn.predict(data)
print('10 fold test:', 100*knn_dict[1])
print('Test accuracy:',100*(la==y_test).mean())
print('Train accuracy:',100*(la2==y_tr).mean())
print('Dataset accuracy:',100*(la3==labels).mean())
print('\n\n---------------------------------------------------------------')


y_pred_svm = svm.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_knn = knn.predict(X_test)
test_acc.append([y_pred_svm,y_pred_rf,y_pred_knn])

a_svm = metrics.accuracy_score(y_test, y_pred_svm)
a_rf = metrics.accuracy_score(y_test, y_pred_rf)
a_knn = metrics.accuracy_score(y_test, y_pred_knn)

print('\n')
print('Test Score for SVM:', a_svm)
print('Test Score for RF:',a_rf)
print('Test Score for KNN:',a_knn)
c_m_svm = metrics.confusion_matrix(y_test, y_pred_svm)
c_m_rf = metrics.confusion_matrix(y_test, y_pred_rf)
c_m_knn = metrics.confusion_matrix(y_test, y_pred_knn)
print('\n')
print('SVM Comfusion Matrix:')
print_cm(c_m_svm,['Relax','Stress'])
print('\n')
print('RF Comfusion Matrix:')
print_cm(c_m_rf,['Relax','Stress'])
print('\n')
print('KNN Comfusion Matrix:')
print_cm(c_m_knn,['Relax','Stress'])

comfusion_matrix.append('SVM physio dataset')
comfusion_matrix.append(c_m_svm)
comfusion_matrix.append('RF physio dataset')
comfusion_matrix.append(c_m_rf)
comfusion_matrix.append('KNN physio dataset')
comfusion_matrix.append(c_m_knn)