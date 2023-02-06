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
from sklearn.metrics import roc_auc_score, plot_confusion_matrix
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

import pdb

from fusion_utils.functions import *

#####################################################################
#####################################################################
#####################################################################
    
baseline = False
trainset_eval = False
hyper_tune = False
cv_k = True

list_nonbaseline = ['datasets/physio_data_new.csv', 'datasets/kinect_data_new.csv', 'datasets/facereader_data_new.csv','datasets/full_feature_dataset.csv']
list_nonbaseline_tr = ['datasets/part_out/physio_train.csv', 'datasets/part_out/kinect_train.csv', 'datasets/part_out/face_train.csv','datasets/part_out/all_feat_train.csv']
list_nonbaseline_test = ['datasets/part_out/physio_test.csv', 'datasets/part_out/kinect_test.csv', 'datasets/part_out/face_test.csv','datasets/part_out/all_feat_test.csv']
list_baseline = ['datasets/baseline_datasets/physio_baseline_data.csv', 'datasets/baseline_datasets/kinect_baseline_data.csv', 'datasets/baseline_datasets/face_baseline_data.csv', 'datasets/baseline_datasets/full_feature_baseline_dataset.csv']

dict_datasets = {"Physio" : 0, "Kinect" : 1, "Face" : 2, "All_features" : 3}



RESULTS = os.path.join(os.getcwd(), "CMs")
os.makedirs(RESULTS, exist_ok=True)

RESULTS_metrics = os.path.join(os.getcwd(), "curves")
os.makedirs(RESULTS_metrics, exist_ok=True)


print('Run_1: Physio data...')
X_tr, X_test, y_tr, y_test, labels, feats = dataset_preprocess(list_nonbaseline[0])
if hyper_tune:
    classifiers = hyperparameter_tuning(X_tr, y_tr, X_test, y_test)
    models = classifiers.models
    clf_svm_physio =  models['SVM'][0]
    clf_rf_physio =  models['Random_Forest'][0]
    clf_knn_physio =  models['KNN'][0]
else:
    print('Start training phase..')
    clf_svm_physio = SVC(kernel='rbf', C=2.51, gamma=0.1, probability=True)
    clf_rf_physio = RandomForestClassifier(n_estimators=700, max_depth=25, warm_start=True, 
                oob_score=True)
    clf_knn_physio = KNeighborsClassifier(n_neighbors = 5)

    if cv_k:
        print("CV10 for SVM")
        cross_mulitclass(clf_svm_physio, X_tr, y_tr, 10)
        print("CV10 for RF")
        cross_mulitclass(clf_rf_physio, X_tr, y_tr, 10)
        print("CV10 for KNN")
        cross_mulitclass(clf_knn_physio, X_tr, y_tr, 10)

    if trainset_eval:
        val_2_trainset([clf_svm_physio, clf_rf_physio, clf_knn_physio], X_tr, y_tr)

clf_svm_physio = clf_svm_physio.fit(X_tr, y_tr)
clf_rf_physio = clf_rf_physio.fit(X_tr, y_tr)
clf_knn_physio = clf_knn_physio.fit(X_tr, y_tr)
    
print("Train data: {}".format(X_tr.shape[0]))
print("Test data: {}".format(X_test.shape[0]))
results_custom(clf_rf_physio, clf_svm_physio, clf_knn_physio, y_test, "Physio", X_test, RESULTS_metrics, RESULTS)

#####################################################################


print('Run_2: Kinect data...')
X_tr, X_test, y_tr, y_test, labels, feats = dataset_preprocess(list_nonbaseline[1])

if hyper_tune:
    classifiers = hyperparameter_tuning(X_tr, y_tr, X_test, y_test)
    models = classifiers.models
    clf_svm_kinect =  models['SVM'][0]
    clf_rf_kinect =  models['Random_Forest'][0]
    clf_knn_kinect =  models['KNN'][0]
else:
    print('Start training phase..')    
    clf_svm_kinect = SVC(kernel='rbf', C=2.81, gamma=0.01, probability=True)
    clf_rf_kinect = RandomForestClassifier(n_estimators=800, max_depth=40, warm_start=True, 
                                            oob_score=True)
    clf_knn_kinect = KNeighborsClassifier(n_neighbors = 1)

    if cv_k:
        print("CV10 for SVM")
        cross_mulitclass(clf_svm_kinect, X_tr, y_tr, 10)
        print("CV10 for RF")
        cross_mulitclass(clf_rf_kinect, X_tr, y_tr, 10)
        print("CV10 for KNN")
        cross_mulitclass(clf_knn_kinect, X_tr, y_tr, 10)

    if trainset_eval:
        val_2_trainset([clf_svm_kinect,clf_rf_kinect,clf_knn_kinect], X_tr, y_tr)
    
    clf_svm_kinect = clf_svm_kinect.fit(X_tr,y_tr)
    clf_rf_kinect = clf_rf_kinect.fit(X_tr,y_tr)
    clf_knn_kinect = clf_knn_kinect.fit(X_tr,y_tr)
print("Train data: {}".format(X_tr.shape[0]))
print("Test data: {}".format(X_test.shape[0]))
results_custom(clf_rf_kinect, clf_svm_kinect, clf_knn_kinect, y_test, "Kinect", X_test, RESULTS_metrics, RESULTS)


#####################################################################


print('Run_3: FaceReader data...')
X_tr, X_test, y_tr, y_test, labels, feats = dataset_preprocess(list_nonbaseline[2])

if hyper_tune:
    classifiers = hyperparameter_tuning(X_tr, y_tr, X_test, y_test)
    models = classifiers.models
    clf_svm_face =  models['SVM'][0]
    clf_rf_face =  models['Random_Forest'][0]
    clf_knn_face =  models['KNN'][0]
else:
    print('Start training phase..')    
    clf_svm_face = SVC(kernel='rbf', C=2.11, gamma=0.1, probability=True)
    clf_rf_face = RandomForestClassifier(n_estimators=400, max_depth=45)
    clf_knn_face = KNeighborsClassifier(n_neighbors = 1)

    if cv_k:
        print("CV10 for SVM")
        cross_mulitclass(clf_svm_face, X_tr, y_tr, 10)
        print("CV10 for RF")
        cross_mulitclass(clf_rf_face, X_tr, y_tr, 10)
        print("CV10 for KNN")
        cross_mulitclass(clf_knn_face, X_tr, y_tr, 10)

    if trainset_eval:
        val_2_trainset([clf_svm_face,clf_rf_face,clf_knn_face], X_tr, y_tr)
    
    ## Train algorithm with train set
    clf_svm_face = clf_svm_face.fit(X_tr,y_tr)
    clf_rf_face = clf_rf_face.fit(X_tr,y_tr)
    clf_knn_face = clf_knn_face.fit(X_tr,y_tr)
print("Train data: {}".format(X_tr.shape[0]))
print("Test data: {}".format(X_test.shape[0]))
results_custom(clf_rf_face, clf_svm_face, clf_knn_face, y_test, "Face", X_test, RESULTS_metrics, RESULTS)


#####################################################################

print('Run_4: All data included...')
X_tr, X_test, y_tr, y_test, labels, feats = dataset_preprocess(list_nonbaseline[3])

if hyper_tune:
    classifiers = hyperparameter_tuning(X_tr, y_tr, X_test, y_test)
    models = classifiers.models
    clf_svm =  models['SVM'][0]
    clf_rf =  models['Random_Forest'][0]
    clf_knn =  models['KNN'][0]
else:
    svm_cl = SVC(kernel='rbf', C=1.81, gamma=0.01, probability=True)
    rf_cl = RandomForestClassifier(n_estimators=1100, max_depth=25)
    knn_cl = KNeighborsClassifier(n_neighbors = 1)

    if trainset_eval:
        val_2_trainset([svm_cl,rf_cl,knn_cl], X_tr, y_tr)
    
    if cv_k:
        print("CV10 for SVM")
        cross_mulitclass(svm_cl, X_tr, y_tr, 10)
        print("CV10 for RF")
        cross_mulitclass(rf_cl, X_tr, y_tr, 10)
        print("CV10 for KNN")
        cross_mulitclass(knn_cl, X_tr, y_tr, 10)

    clf_svm = svm_cl.fit(X_tr,y_tr)
    clf_rf = rf_cl.fit(X_tr,y_tr)
    clf_knn = knn_cl.fit(X_tr,y_tr)
print("Train data: {}".format(X_tr.shape[0]))
print("Test data: {}".format(X_test.shape[0]))
results_custom(clf_rf, clf_svm, clf_knn, y_test, "Fusion", X_test, RESULTS_metrics, RESULTS)









#####################################################################
## FUSION HELPING FUNCTIONS

def pred(clf, data):
    zz = np.zeros((2, data.shape[0]))
    y_pred = clf.predict(data)
    y_prob = clf.predict_proba(data)
    zz[0, :] = y_pred.astype(np.int)
    zz[1, :] = np.max(y_prob, axis=1)
    return zz

def pred_NN(data, modelname):
    zz = np.zeros((2, data.shape[0]))
    model = Model(data.shape[1], 256, 2)
    model = torch.load(modelname)
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











########################################################
############# FUSION SENSOR ARCHITECTUERS ##############
########################################################


print('Sensor fusion...')
X_physio, X_kinect, X_face, y_labels  = dataset_preprocess_fusion()

d1 = pred(clf_rf_physio, X_physio)
d2 = pred(clf_knn_kinect, X_kinect)
d3 = pred(clf_svm_face, X_face)

model_physio_path = "ANN/models/demo_presentation/physio/model_ANN_physio.pt"
model_kinect_path = "ANN/models/demo_presentation/kinect/model_ANN_kinect.pt"
model_facial_path = "ANN/models/demo_presentation/face/model_ANN_face.pt"
d1_NN = pred_NN(X_physio, model_physio_path)
d2_NN = pred_NN(X_kinect, model_kinect_path)
d3_NN = pred_NN(X_face, model_facial_path)





#########################################################
######### MAJORITY FUSION SENSOR ARCHITECTUERS ##########
#########################################################

decision_system_classification, mds_labels = \
                    sensor_fusion_decision(d1, d2, d3, "Majority")
dsf = np.array(decision_system_classification)
print(metrics.classification_report(y_labels, dsf))


c_m_dsf = metrics.confusion_matrix(y_labels, dsf)
print('Sonsor Decision Fusion Comfusion Matrix:')
print_cm(c_m_dsf,['Relax','Stress'])
    
tn, fp, fn, tp = c_m_dsf.ravel()
print('\n')
print("Sensor Fusion")
print('F1 Score:', round(100*float((2*tp)/(2*tp+fn+fp)),6),'%')
print('Precision:', round(100*float(tp/(tp+fp)),6),'%')
print('Recall:', round(100*float(tp/(tp+fn)),6),'%')
print('Accuracy:', round(100*float((tp+tn)/(tp+tn+fn+fp)),6),'%')


decision_system_classification, mds_labels = \
                    sensor_fusion_decision(d1, d2_NN, d3_NN, "Majority")
dsf = np.array(decision_system_classification)
print(metrics.classification_report(y_labels, dsf))


c_m_dsf = metrics.confusion_matrix(y_labels, dsf)
print('Sonsor Decision Fusion Comfusion Matrix:')
print_cm(c_m_dsf,['Relax','Stress'])
    
tn, fp, fn, tp = c_m_dsf.ravel()
print('\n')
print("Sensor Fusion - Some NNs")
print('F1 Score:', round(100*float((2*tp)/(2*tp+fn+fp)),6),'%')
print('Precision:', round(100*float(tp/(tp+fp)),6),'%')
print('Recall:', round(100*float(tp/(tp+fn)),6),'%')
print('Accuracy:', round(100*float((tp+tn)/(tp+tn+fn+fp)),6),'%')



decision_system_classification, mds_labels = \
                    sensor_fusion_decision(d1_NN, d2_NN, d3_NN, "Majority")
dsf = np.array(decision_system_classification)
print(metrics.classification_report(y_labels, dsf))


c_m_dsf = metrics.confusion_matrix(y_labels, dsf)
print('Sonsor Decision Fusion Comfusion Matrix:')
print_cm(c_m_dsf,['Relax','Stress'])
    
tn, fp, fn, tp = c_m_dsf.ravel()
print('\n')
print("Sensor Fusion - All NNs")
print('F1 Score:', round(100*float((2*tp)/(2*tp+fn+fp)),6),'%')
print('Precision:', round(100*float(tp/(tp+fp)),6),'%')
print('Recall:', round(100*float(tp/(tp+fn)),6),'%')
print('Accuracy:', round(100*float((tp+tn)/(tp+tn+fn+fp)),6),'%')






#########################################################
######### WEIGHTED FUSION SENSOR ARCHITECTUERS ##########
#########################################################



decision_system_classification, mds_labels = sensor_fusion_decision(d1, d2, d3, "weighted-majority")
dsf = np.array(decision_system_classification)
print(metrics.classification_report(y_labels, dsf))

c_m_dsf = metrics.confusion_matrix(y_labels, dsf)
print('Sonsor Decision Fusion Comfusion Matrix Weighted Majority:')
print_cm(c_m_dsf,['Relax','Stress'])
    
tn, fp, fn, tp = c_m_dsf.ravel()
print('\n')
print("Sensor Fusion - Weighted Majority")
print('F1 Score:', round(100*float((2*tp)/(2*tp+fn+fp)),6),'%')
print('Precision:', round(100*float(tp/(tp+fp)),6),'%')
print('Recall:', round(100*float(tp/(tp+fn)),6),'%')
print('Accuracy:', round(100*float((tp+tn)/(tp+tn+fn+fp)),6),'%')


decision_system_classification, mds_labels = \
    sensor_fusion_decision(d1, d2_NN, d3_NN, "weighted-majority")
dsf = np.array(decision_system_classification)
print(metrics.classification_report(y_labels, dsf))

c_m_dsf = metrics.confusion_matrix(y_labels, dsf)
print('Sonsor Decision Fusion Comfusion Matrix Weighted Majority:')
print_cm(c_m_dsf,['Relax','Stress'])
    
tn, fp, fn, tp = c_m_dsf.ravel()
print('\n')
print("Sensor Fusion - Some NNs- Weighted Majority")
print('F1 Score:', round(100*float((2*tp)/(2*tp+fn+fp)),6),'%')
print('Precision:', round(100*float(tp/(tp+fp)),6),'%')
print('Recall:', round(100*float(tp/(tp+fn)),6),'%')
print('Accuracy:', round(100*float((tp+tn)/(tp+tn+fn+fp)),6),'%')


decision_system_classification, mds_labels = \
    sensor_fusion_decision(d1_NN, d2_NN, d3_NN, "weighted-majority")
dsf = np.array(decision_system_classification)
print(metrics.classification_report(y_labels, dsf))

c_m_dsf = metrics.confusion_matrix(y_labels, dsf)
print('Sensor Decision Fusion Comfusion Matrix Weighted Majority:')
print_cm(c_m_dsf,['Relax','Stress'])
    
tn, fp, fn, tp = c_m_dsf.ravel()
print('\n')
print("Sensor Fusion - Weighted Majority - All NN")
print('F1 Score:', round(100*float((2*tp)/(2*tp+fn+fp)),6),'%')
print('Precision:', round(100*float(tp/(tp+fp)),6),'%')
print('Recall:', round(100*float(tp/(tp+fn)),6),'%')
print('Accuracy:', round(100*float((tp+tn)/(tp+tn+fn+fp)),6),'%')


