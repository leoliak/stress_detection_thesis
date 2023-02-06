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

#####################################################################
#####################################################################
#####################################################################

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
        self.layer_1_BN = nn.BatchNorm1d(hidden_size*5)
        self.RELU1 = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size*5, hidden_size*3)
        self.layer_2_BN = nn.BatchNorm1d(hidden_size*3)
        self.RELU2 = nn.ReLU()
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
    print("\n\n")


def prediction_metrics(model, test_data, test_labels, name):
    #accuracy
    print('{} test accuracy = {}\n'.format(name, 100*(model.predict(test_data) == test_labels).mean()))
    #AUC of ROC
    prob = model.predict_proba(test_data)
    auc=roc_auc_score(test_labels.reshape(-1),prob[:,1])
    auc=roc_auc_score(test_labels,prob[:,1])
    print('Classifier {} area under curve of ROC is {}\n'.format(name,100*auc))
    #ROC
    fpr, tpr, thresholds = roc_curve(test_labels.reshape(-1), prob[:,1], pos_label=1)
    roc_plot(fpr,tpr,name,auc)


def roc_plot(fpr, tpr ,name, auc):
    plt.figure(figsize=(20,5))
    plt.plot(fpr,tpr)
    plt.ylim([0.0,1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC of {} AUC: {}\n'.format(name, auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.show()


class Classifiers(object):
    def __init__(self, train_data, train_labels, models, hyperTune=True):
        self.train_data=train_data
        self.train_labels=train_labels
        self.models = models
        self.construct_all_models(hyperTune)
        
    
    def construct_all_models(self, hyperTune):
        if hyperTune:
            for name, candidate_hyperParam in self.models.items():
                self.models[name] = self.train_with_hyperParamTuning(candidate_hyperParam[0], name,candidate_hyperParam[1])
            print ('\nTraining process finished\n')
          
            
    def train_with_hyperParamTuning(self, model, name, param_grid):
        #grid search method for hyper-parameter tuning
        grid = GridSearchCV(model, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
        grid.fit(self.train_data, self.train_labels)
        print(
            '\nThe best hyper-parameter for  {} is {}, mean accuracy through 10 Fold test is {} \n'\
            .format(name, grid.best_params_, round(100*grid.best_score_,2)))

        model = grid.best_estimator_
        score = grid.best_score_
        parameters = grid.best_params_
        train_pred = model.predict(self.train_data)
        print('{} train accuracy = {}\n'.format(name,100*(train_pred == self.train_labels).mean()))
        return model, score, parameters



def visualize_comfusion_matrix(classifier, X_test, y_test, class_names, print_name):
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    titles_options = [(print_name + " Confusion matrix", None),
                      (print_name + " Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test,
                                    display_labels=class_names,
                                    cmap=plt.cm.Blues,
                                    normalize=normalize)
        disp.ax_.set_title(title)
        plt.savefig(RESULTS + "/" + title + "_" + print_name.replace(' ', '_') + ".png")
        plt.close()
    

def results(clf_rf, clf_svm, clf_knn, y_test, labels, feats, exp_name, X_test):  
    # list1, list2 = rmv_non_important_features(clf_rf, feats)

    y_pred_svm = clf_svm.predict(X_test)
    y_pred_rf = clf_rf.predict(X_test)
    y_pred_knn = clf_knn.predict(X_test)

    ## Calculate Classifiers metrics
    cr_svm = metrics.classification_report(y_test,y_pred_svm, output_dict=True)
    cr_rf = metrics.classification_report(y_test, y_pred_rf, output_dict=True)
    cr_knn = metrics.classification_report(y_test,y_pred_knn, output_dict=True)

    print('\n')
    print('F1 Score for SVM:', round(100*cr_svm["weighted avg"]["f1-score"],6),'%')
    print('F1 Score for RF:',round(100*cr_rf["weighted avg"]["f1-score"],6),'%')
    print('F1 Score for KNN:',round(100*cr_knn["weighted avg"]["f1-score"],6),'%')
    print('\n')
    print('Precision for SVM:', round(100*cr_svm["weighted avg"]["precision"],6),'%')
    print('Precision for RF:',round(100*cr_rf["weighted avg"]["precision"],6),'%')
    print('Precision for KNN:',round(100*cr_knn["weighted avg"]["precision"],6),'%')
    print('\n')
    print('Recall for SVM:', round(100*cr_svm["weighted avg"]["recall"],6),'%')
    print('Recall for RF:',round(100*cr_rf["weighted avg"]["recall"],6),'%')
    print('Recall for KNN:',round(100*cr_knn["weighted avg"]["recall"],6),'%')
    print('\n')
    print('Test Score for SVM:', round(100*cr_svm["accuracy"],6),'%')
    print('Test Score for RF:',round(100*cr_rf["accuracy"],6),'%')
    print('Test Score for KNN:',round(100*cr_knn["accuracy"],6),'%')

    ## Confusion Matrix
    c_m_svm = metrics.confusion_matrix(y_test, y_pred_svm)
    c_m_rf = metrics.confusion_matrix(y_test, y_pred_rf)
    c_m_knn = metrics.confusion_matrix(y_test, y_pred_knn)

    print('\n')
    print('SVM Comfusion Matrix:')
    print_cm(c_m_svm,['Relax','Stress'])
    visualize_comfusion_matrix(clf_svm, X_test, y_test, ['Relax','Stress'], exp_name + " SVM")

    print('RF Comfusion Matrix:')
    print_cm(c_m_rf,['Relax','Stress'])
    visualize_comfusion_matrix(clf_rf, X_test, y_test, ['Relax','Stress'], exp_name + " RF")

    print('KNN Comfusion Matrix:')
    print_cm(c_m_knn,['Relax','Stress'])
    visualize_comfusion_matrix(clf_knn, X_test, y_test, ['Relax','Stress'], exp_name + " KNN")
    
    unique, counts = np.unique(y_test, return_counts=True)
    dic = dict(zip(unique, counts))
    unique, counts = np.unique(labels, return_counts=True)
    dic2 = dict(zip(unique, counts))




def roc(fpr, tpr ,exp_name,  name):
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(RESULTS_metrics + "/ROC_" + exp_name + "_" + name + ".png")
    plt.close()

def curves(clf, y_score, y_test, acc, exp_name, clname):
    ## ROC
    fpr, tpr, _ = metrics.roc_curve(y_test, y_score,
                                         pos_label=clf.classes_[1])
    roc_display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc = metrics.auc(fpr, tpr), estimator_name=clname).plot()

    ## Precision-Recall
    prec, recall, _ = metrics.precision_recall_curve(y_test, y_score,
                                         pos_label=clf.classes_[1])
    pr_display = metrics.PrecisionRecallDisplay(precision=prec, recall=recall,
                                                average_precision=acc, estimator_name=clname).plot()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    roc_display.plot(ax=ax1)
    pr_display.plot(ax=ax2)
    plt.savefig(RESULTS_metrics + "/Curves" + exp_name + "_" + clname + ".png")
    plt.close()

def curves_2(clf, y_test, exp_name, clname):
    ## ROC
    disp_prc = metrics.plot_precision_recall_curve(clf, X_test, y_test)
    disp_roc = metrics.plot_roc_curve(clf, X_test, y_test)
    ## Precision-Recall
    plt.figure()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    disp_roc.plot(ax=ax1)
    disp_prc.plot(ax=ax2)
    disp_roc.ax_.set_title('ROC curve')
    disp_prc.ax_.set_title('Precision-Recall curve')
    plt.savefig(RESULTS_metrics + "/Curves" + exp_name + "_" + clname + ".png")
    plt.close()



def results_custom(clf_rf, clf_svm, clf_knn, y_test, labels, feats, exp_name, X_test):  
    # list1, list2 = rmv_non_important_features(clf_rf, feats)

    y_pred_svm = clf_svm.predict(X_test)
    y_pred_rf = clf_rf.predict(X_test)
    y_pred_knn = clf_knn.predict(X_test)

    ## Confusion Matrix
    c_m_svm = metrics.confusion_matrix(y_test, y_pred_svm)
    c_m_rf = metrics.confusion_matrix(y_test, y_pred_rf)
    c_m_knn = metrics.confusion_matrix(y_test, y_pred_knn)

    print('\n')
    print('SVM Comfusion Matrix:')
    print_cm(c_m_svm,['Relax','Stress'])
    visualize_comfusion_matrix(clf_svm, X_test, y_test, ['Relax','Stress'], exp_name + " SVM")

    print('RF Comfusion Matrix:')
    print_cm(c_m_rf,['Relax','Stress'])
    visualize_comfusion_matrix(clf_rf, X_test, y_test, ['Relax','Stress'], exp_name + " RF")

    print('KNN Comfusion Matrix:')
    print_cm(c_m_knn,['Relax','Stress'])
    visualize_comfusion_matrix(clf_knn, X_test, y_test, ['Relax','Stress'], exp_name + " KNN")
        
    tn, fp, fn, tp = c_m_svm.ravel()
    print('\n')
    print("SVM")
    print('F1 Score:', round(100*float((2*tp)/(2*tp+fn+fp)),6),'%')
    print('Precision:', round(100*float(tp/(tp+fp)),6),'%')
    print('Recall:', round(100*float(tp/(tp+fn)),6),'%')
    print('Accuracy:', round(100*float((tp+tn)/(tp+tn+fn+fp)),6),'%')
    curves_2(clf_svm, y_test, exp_name, "SVM")


    tn, fp, fn, tp = c_m_rf.ravel()
    print('\n')
    print("Random Forest")
    print('F1 Score:',round(100*float((2*tp)/(2*tp+fn+fp)),6),'%')
    print('Precision:',round(100*float(tp/(tp+fp)),6),'%')
    print('Recall:',round(100*float(tp/(tp+fn)),6),'%')
    print('Accuracy:', round(100*float((tp+tn)/(tp+tn+fn+fp)),6),'%')
    curves_2(clf_rf, y_test, exp_name, "RF")


    tn, fp, fn, tp = c_m_knn.ravel()
    print('\n')
    print('F1 Score:',round(100*float((2*tp)/(2*tp+fn+fp)),6),'%')
    print('Precision:',round(100*float(tp/(tp+fp)),6),'%')
    print('Recall:',round(100*float(tp/(tp  +fn)),6),'%')
    print('Accuracy:', round(100*float((tp+tn)/(tp+tn+fn+fp)),6),'%')
    curves_2(clf_knn, y_test, exp_name, "KNN")
    print('\n')






def rmv_non_important_features(clf_rf, feats, imp_thres = 0.004):
    # get importance
    feat_importance = clf_rf.feature_importances_

    # Print feature importance
    for i,v in enumerate(feat_importance):
        print('Feature: {}, Score: {}'.format(feats[i], v))
    
    x = dict(zip(feats, feat_importance))
    sorted_x = sorted(x.items(), key=lambda kv: kv[1],reverse=True)
    li, li_ = [], []
    for (key, value) in x.items():
        if value>=imp_thres:
           li.append(key)
        else:
            li_.append(key)
    return li, li_




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
    # df = df.drop(['SD1','SD2','SDSD'],axis=1)
    df = df.fillna(df.mean())
    df = df.replace([-np.inf], 0.0)
    feats = df.columns

    X = df.values
    # feature extraction
    pca = PCA(n_components=int(df.shape[1]*0.4))
    pca.fit(X)
    X1 = pca.transform(X)

    st_scaler.fit(df)
    data = st_scaler.transform(df)   
    X_tr, X_test, y_tr, y_test = train_test_split(data, labels, test_size = 0.2, random_state = 42)
    
    unique, counts = np.unique(y_tr, return_counts=True)
    print("Train set")
    print(dict(zip(unique, counts)))
    unique, counts = np.unique(y_test, return_counts=True)
    print("Test set")
    print(dict(zip(unique, counts)))
    return X_tr, X_test, y_tr, y_test, labels, feats
      

def val_2_trainset(class_list, X_tr, Y_tr):
    scores_svm = cross_val_score(class_list[0], X_tr, y_tr, cv=10)
    scores_rf = cross_val_score(class_list[1], X_tr, y_tr, cv=10)
    scores_knn = cross_val_score(class_list[2], X_tr, y_tr, cv=10)
    
    print('10-fold Score for SVM on train set:', round(100*scores_svm.mean(),4),'%')
    print('10-fold Score for RF on train set:',round(100*scores_rf.mean(),4),'%')
    print('10-fold Score for KNN on train set:',round(100*scores_knn.mean(),4),'%')
    

def hyperparameter_tuning(X_tr, y_tr, X_test, y_test):
    ## Define models for hyperparameter optimization
    models = {
                'SVM':[SVC(probability=True),dict(kernel=['rbf'],gamma=np.logspace(-3, 1, 5),C=np.arange(0.01, 3.01, 0.2))],
                'KNN':[KNeighborsClassifier(),dict(n_neighbors=np.arange(1, 20))],
                'Random_Forest':[RandomForestClassifier(),dict(n_estimators = np.arange(200,1300,100),max_depth = np.linspace(20, 50, 7, endpoint=True))]
             }
    print('Start training phase..')
    classifiers = Classifiers(X_tr, y_tr, models, True)
    return classifiers



def dataset_preprocess_fusion():
    print("Load and preprocess all data for sesor fusion experiment")
    datas_i = []
    label_i = []
    di = {}
    baseline = False
    df_name = dict_datasets["All_features"]
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
    df_t = df_t.sample(frac=1).reset_index(drop=True)
    labels = df_t["Condition"]
    print(labels.value_counts())
    df = df_t.drop(df_t.columns[[0,1,2,3,4]],axis=1)
    # df = df.drop(['SD1','SD2','SDSD'],axis=1)
    df = df.fillna(df.mean())
    df = df.replace([-np.inf], 0.0)
    st_scaler.fit(df)
    data = st_scaler.transform(df)
    X_physio = data[:, 0:35]
    X_kinect = data[:, 35:129]
    X_face = data[:, 129:]
    y_labels = labels.values
    
    return X_physio[-450:, :], X_kinect[-450:, :], X_face[-450:, :], y_labels[-450:]

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


#####################################################################
#####################################################################
#####################################################################


if __name__ == "__main__":
    
    baseline = False
    trainset_eval = False
    hyper_tune = False

    list_nonbaseline = ['datasets/physio_data_new.csv', 'datasets/kinect_data_new.csv', 'datasets/facereader_data_new.csv','datasets/full_feature_dataset.csv']
    list_baseline = ['baseline_datasets/physio_baselinne_data.csv', 'baseline_datasets/kinect_baseline_data.csv', 'baseline_datasets/face_baseline_data.csv', 'baseline_datasets/full_feature_baseline_dataset.csv']

    dict_datasets = {"Physio" : 0, "Kinect" : 1, "Face" : 2, "All_features" : 3}

    RESULTS = os.path.join(os.getcwd(), "CMs")
    os.makedirs(RESULTS, exist_ok=True)

    RESULTS_metrics = os.path.join(os.getcwd(), "curves")
    os.makedirs(RESULTS_metrics, exist_ok=True)

    print('Run_1: Physio data...')
    X_tr, X_test, y_tr, y_test, labels, feats = dataset_preprocess("Physio", baseline)

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
        if trainset_eval:
            val_2_trainset([clf_svm_physio, clf_rf_physio, clf_knn_physio], X_tr, y_tr)
        clf_svm_physio = clf_svm_physio.fit(X_tr,y_tr)
        clf_rf_physio = clf_rf_physio.fit(X_tr,y_tr)
        clf_knn_physio = clf_knn_physio.fit(X_tr,y_tr)
    print("Train data: {}".format(X_tr.shape[0]))
    print("Test data: {}".format(X_test.shape[0]))
    results_custom(clf_rf_physio, clf_svm_physio, clf_knn_physio, y_test, labels, feats, "Physio", X_test)

    #####################################################################


    print('Run_2: Kinect data...')
    X_tr, X_test, y_tr, y_test, labels, feats = dataset_preprocess("Kinect", baseline)

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

        if trainset_eval:
            val_2_trainset([clf_svm_kinect,clf_rf_kinect,clf_knn_kinect], X_tr, y_tr)
        
        clf_svm_kinect = clf_svm_kinect.fit(X_tr,y_tr)
        clf_rf_kinect = clf_rf_kinect.fit(X_tr,y_tr)
        clf_knn_kinect = clf_knn_kinect.fit(X_tr,y_tr)
    print("Train data: {}".format(X_tr.shape[0]))
    print("Test data: {}".format(X_test.shape[0]))
    results_custom(clf_rf_kinect, clf_svm_kinect, clf_knn_kinect, y_test, labels, feats, "Kinect", X_test)


    #####################################################################


    print('Run_3: FaceReader data...')
    X_tr, X_test, y_tr, y_test, labels, feats = dataset_preprocess("Face", baseline)

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

        if trainset_eval:
            val_2_trainset([clf_svm_face,clf_rf_face,clf_knn_face], X_tr, y_tr)
        
        ## Train algorithm with train set
        clf_svm_face = clf_svm_face.fit(X_tr,y_tr)
        clf_rf_face = clf_rf_face.fit(X_tr,y_tr)
        clf_knn_face = clf_knn_face.fit(X_tr,y_tr)
    print("Train data: {}".format(X_tr.shape[0]))
    print("Test data: {}".format(X_test.shape[0]))
    results_custom(clf_rf_face, clf_svm_face, clf_knn_face, y_test, labels, feats, "Face", X_test)


    #####################################################################
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


    print('Sensor fusion...')
    X_physio, X_kinect, X_face, y_labels  = dataset_preprocess_fusion()

    d1 = pred(clf_rf_physio, X_physio)
    d2 = pred(clf_knn_kinect, X_kinect)
    d3 = pred(clf_svm_face, X_face)

    d1_NN = pred_NN(X_physio, "model_ANN_v2_physio.pt")
    d2_NN = pred_NN(X_kinect, "model_ANN_v2_kinect.pt")
    d3_NN = pred_NN(X_face, "model_ANN_v2_face.pt")


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


## NN
