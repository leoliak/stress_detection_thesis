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

trainset_eval = False
hyper_tune = False
cv_k = True

list_nonbaseline_tr = ['datasets/part_out/physio_train.csv', 'datasets/part_out/kinect_train.csv', 'datasets/part_out/face_train.csv','datasets/part_out/all_feat_train.csv']
list_nonbaseline_test = ['datasets/part_out/physio_test.csv', 'datasets/part_out/kinect_test.csv', 'datasets/part_out/face_test.csv','datasets/part_out/all_feat_test.csv']

list_nonbaseline = ['datasets/physio_data_new.csv', 'datasets/kinect_data_new.csv', 'datasets/facereader_data_new.csv','datasets/full_feature_dataset.csv']
list_baseline = ['datasets/baseline_datasets/physio_baseline_data.csv', 'datasets/baseline_datasets/kinect_baseline_data.csv', 'datasets/baseline_datasets/face_baseline_data.csv', 'datasets/baseline_datasets/full_feature_baseline_dataset.csv']

dict_datasets = {"Physio" : 0, "Kinect" : 1, "Face" : 2, "All_features" : 3}

RESULTS = os.path.join(os.getcwd(), "CMs")
os.makedirs(RESULTS, exist_ok=True)

RESULTS_metrics = os.path.join(os.getcwd(), "curves")
os.makedirs(RESULTS_metrics, exist_ok=True)

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



def visualize_comfusion_matrix(classifier, X_test, y_test, class_names, print_name, RESULTS):
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
    

def results(clf_rf, clf_svm, clf_knn, y_test, labels, feats, exp_name, X_test, RESULTS):  
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
    visualize_comfusion_matrix(clf_svm, X_test, y_test, ['Relax','Stress'], exp_name + " SVM", RESULTS)

    print('RF Comfusion Matrix:')
    print_cm(c_m_rf,['Relax','Stress'])
    visualize_comfusion_matrix(clf_rf, X_test, y_test, ['Relax','Stress'], exp_name + " RF", RESULTS)

    print('KNN Comfusion Matrix:')
    print_cm(c_m_knn,['Relax','Stress'])
    visualize_comfusion_matrix(clf_knn, X_test, y_test, ['Relax','Stress'], exp_name + " KNN", RESULTS)
    
    unique, counts = np.unique(y_test, return_counts=True)
    dic = dict(zip(unique, counts))
    unique, counts = np.unique(labels, return_counts=True)
    dic2 = dict(zip(unique, counts))




def roc(fpr, tpr ,exp_name, name, RESULTS_metrics):
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

def curves(clf, y_score, y_test, acc, exp_name, clname, RESULTS_metrics):
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

def curves_2(clf, X_test, y_test, exp_name, clname, RESULTS_metrics):
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



def results_custom(clf_rf, clf_svm, clf_knn, y_test, exp_name, X_test, RESULTS_metrics, RESULTS):  
    # list1, list2 = rmv_non_important_features(clf_rf, feats)

    di = {
        'SVM': {
                'f1_score': 0,
                'precision': 0,
                'recall': 0,
                'accuracy': 0,
               },
        'RF': {
                'f1_score': 0,
                'precision': 0,
                'recall': 0,
                'accuracy': 0,
               },
        'KNN': {
                'f1_score': 0,
                'precision': 0,
                'recall': 0,
                'accuracy': 0,
               }
    }
    y_pred_svm = clf_svm.predict(X_test)
    y_pred_rf = clf_rf.predict(X_test)
    y_pred_knn = clf_knn.predict(X_test)

    ## Confusion Matrix
    c_m_svm = metrics.confusion_matrix(y_test, y_pred_svm)
    c_m_rf = metrics.confusion_matrix(y_test, y_pred_rf)
    c_m_knn = metrics.confusion_matrix(y_test, y_pred_knn)

    print('\n')
    print('SVM Comfusion Matrix:')
    print(c_m_svm)
    print_cm(c_m_svm,['Relax','Stress'])
    visualize_comfusion_matrix(clf_svm, X_test, y_test, ['Relax','Stress'], exp_name + " SVM", RESULTS)

    print('RF Comfusion Matrix:')
    print(c_m_rf)
    print_cm(c_m_rf,['Relax','Stress'])
    visualize_comfusion_matrix(clf_rf, X_test, y_test, ['Relax','Stress'], exp_name + " RF", RESULTS)

    print('KNN Comfusion Matrix:')
    print(c_m_knn)
    print_cm(c_m_knn,['Relax','Stress'])
    visualize_comfusion_matrix(clf_knn, X_test, y_test, ['Relax','Stress'], exp_name + " KNN", RESULTS)
        
    tn, fp, fn, tp = c_m_svm.ravel()
    print('\n')
    print("SVM")
    print('F1 Score:', round(100*float((2*tp)/(2*tp+fn+fp)),6),'%')
    print('Precision:', round(100*float(tp/(tp+fp)),6),'%')
    print('Recall:', round(100*float(tp/(tp+fn)),6),'%')
    print('Accuracy:', round(100*float((tp+tn)/(tp+tn+fn+fp)),6),'%')
    curves_2(clf_svm, X_test, y_test, exp_name, "SVM", RESULTS_metrics)
    
    di["SVM"]["f1_score"] = round(100*float((2*tp)/(2*tp+fn+fp)),6)
    di["SVM"]["precision"] = round(100*float(tp/(tp+fp)),6)
    di["SVM"]["recall"] = round(100*float(tp/(tp+fn)),6)
    di["SVM"]["accuracy"] = round(100*float((tp+tn)/(tp+tn+fn+fp)),6)


    tn, fp, fn, tp = c_m_rf.ravel()
    print('\n')
    print("Random Forest")
    print('F1 Score:',round(100*float((2*tp)/(2*tp+fn+fp)),6),'%')
    print('Precision:',round(100*float(tp/(tp+fp)),6),'%')
    print('Recall:',round(100*float(tp/(tp+fn)),6),'%')
    print('Accuracy:', round(100*float((tp+tn)/(tp+tn+fn+fp)),6),'%')
    curves_2(clf_rf, X_test, y_test, exp_name, "RF", RESULTS_metrics)

    di["RF"]["f1_score"] = round(100*float((2*tp)/(2*tp+fn+fp)),6)
    di["RF"]["precision"] = round(100*float(tp/(tp+fp)),6)
    di["RF"]["recall"] = round(100*float(tp/(tp+fn)),6)
    di["RF"]["accuracy"] = round(100*float((tp+tn)/(tp+tn+fn+fp)),6)

    tn, fp, fn, tp = c_m_knn.ravel()
    print('\n')
    print('F1 Score:',round(100*float((2*tp)/(2*tp+fn+fp)),6),'%')
    print('Precision:',round(100*float(tp/(tp+fp)),6),'%')
    print('Recall:',round(100*float(tp/(tp  +fn)),6),'%')
    print('Accuracy:', round(100*float((tp+tn)/(tp+tn+fn+fp)),6),'%')
    curves_2(clf_knn,X_test, y_test, exp_name, "KNN", RESULTS_metrics)
    print('\n')

    di["KNN"]["f1_score"] = round(100*float((2*tp)/(2*tp+fn+fp)),6)
    di["KNN"]["precision"] = round(100*float(tp/(tp+fp)),6)
    di["KNN"]["recall"] = round(100*float(tp/(tp+fn)),6)
    di["KNN"]["accuracy"] = round(100*float((tp+tn)/(tp+tn+fn+fp)),6)

    return di







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



# def prep(df_t):
    # print(df_time.columns)
    # st_scaler = preprocessing.StandardScaler()
    
    # labels = df_t["Condition"]
    # print(labels.value_counts())
    # df = df_t.drop(df_t.columns[[0,1,2,3,4]],axis=1)
    # # df = df.drop(['SD1','SD2','SDSD'],axis=1)
    # df = df.fillna(df.mean())
    # df = df.replace([-np.inf], 0.0)
    # feats = df.columns

    # X = df.values

    # st_scaler.fit(df)
    # data = st_scaler.transform(df)   
    # X_, Y_ = data, labels
    
    # return X_, Y_, labels, feats



# def dataset_preprocess(id_name, baseline):
    # print("Load and preprocess %s data.." %id_name)
    # df_name = dict_datasets[id_name]

    # if not baseline:
    #     df_time = pd.read_csv(list_nonbaseline_tr[df_name])
    #     voc = {"Condition": {"N":0, "I":1, "T":1, "R":0}}
    #     df_time1 = pd.read_csv(list_nonbaseline_test[df_name])

    # # print(df_time.columns
    # df_t = df_time
    # df_t.replace(voc, inplace=True)
    # X_tr, y_tr, _, _ =  prep(df_t)

    # df_t = df_time1
    # df_t.replace(voc, inplace=True)
    # X_test, y_test, labels, feats =  prep(df_t)
    
    # unique, counts = np.unique(y_tr, return_counts=True)
    # print("Train set")
    # print(dict(zip(unique, counts)))
    # unique, counts = np.unique(y_test, return_counts=True)
    # print("Test set")
    # print(dict(zip(unique, counts)))

    # return X_tr, X_test, y_tr, y_test, labels, feats
      



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
    return X_tr, X_test, y_tr, y_test, labels, feats



def val_2_trainset(class_list, X_tr, Y_tr):
    scores_svm = cross_val_score(class_list[0], X_tr, Y_tr, cv=10)
    scores_rf = cross_val_score(class_list[1], X_tr, Y_tr, cv=10)
    scores_knn = cross_val_score(class_list[2], X_tr, Y_tr, cv=10)
    
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



def dataset_preprocess_fusion(pp_out = -1):
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
    df = df.drop(['ln_vlf'], axis=1)
    # df = df.drop(['SD1','SD2','SDSD'],axis=1)
    df = df.fillna(df.mean())
    df = df.replace([-np.inf], 0.0)

    dd = df_t.loc[df_t["PP"] == pp_out]
    ind_d = dd.index.tolist()
    fd = ind_d[0]
    ld = ind_d[-1]

    X_test = data[fd:ld, :]
    y_test = labels[fd:ld]

    X_tr = np.concatenate((data[0:fd,:], data[ld:, :]), axis=0)
    y_tr = np.concatenate((labels[0:fd], labels[ld:]), axis=0)
    st_scaler.fit(df)
    data = st_scaler.transform(df)
    lim_1 = 34
    lim_2 = 128
    X_physio = data[:, 0:lim_1]
    X_kinect = data[:, lim_1:lim_2]
    X_face = data[:, lim_2:]
    y_labels = labels.values
    
    return X_physio[-450:, :], X_kinect[-450:, :], X_face[-450:, :], y_labels[-450:]



def dataset_preprocess_fusion_2(pp_out = -1):
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

