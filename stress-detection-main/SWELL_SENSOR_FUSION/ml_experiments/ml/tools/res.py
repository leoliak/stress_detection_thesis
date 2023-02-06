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
    return scores

#####################################################################
#####################################################################
#####################################################################


if __name__ == "__main__":
    
    baseline = False
    trainset_eval = False
    hyper_tune = False
    cv_k = True

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
        clf_svm =  models['SVM'][0]
        clf_rf =  models['Random_Forest'][0]
        clf_knn =  models['KNN'][0]
    else:
        print('Start training phase..')
        svm_cl = SVC(kernel='rbf', C=2.51, gamma=0.1, probability=True)
        rf_cl = RandomForestClassifier(n_estimators=700, max_depth=25, warm_start=True, 
                  oob_score=True)
        knn_cl = KNeighborsClassifier(n_neighbors = 5)
        if trainset_eval:
            val_2_trainset([svm_cl, rf_cl, knn_cl], X_tr, y_tr)
        
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
    results_custom(clf_rf, clf_svm, clf_knn, y_test, labels, feats, "Physio", X_test)

    #####################################################################


    print('Run_2: Kinect data...')
    X_tr, X_test, y_tr, y_test, labels, feats = dataset_preprocess("Kinect", baseline)

    if hyper_tune:
        classifiers = hyperparameter_tuning(X_tr, y_tr, X_test, y_test)
        models = classifiers.models
        svm_cl =  models['SVM'][0]
        clf_rf =  models['Random_Forest'][0]
        clf_knn =  models['KNN'][0]
    else:
        print('Start training phase..')    
        clf_svm = SVC(kernel='rbf', C=2.81, gamma=0.01, probability=True)
        rf_cl = RandomForestClassifier(n_estimators=800, max_depth=40, warm_start=True, 
                  oob_score=True)
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
    results_custom(clf_rf, clf_svm, clf_knn, y_test, labels, feats, "Kinect", X_test)


    #####################################################################


    print('Run_3: FaceReader data...')
    X_tr, X_test, y_tr, y_test, labels, feats = dataset_preprocess("Face", baseline)

    if hyper_tune:
        classifiers = hyperparameter_tuning(X_tr, y_tr, X_test, y_test)
        models = classifiers.models
        clf_svm =  models['SVM'][0]
        clf_rf =  models['Random_Forest'][0]
        clf_knn =  models['KNN'][0]
    else:
        print('Start training phase..')    
        svm_cl = SVC(kernel='rbf', C=2.11, gamma=0.1, probability=True)
        rf_cl = RandomForestClassifier(n_estimators=400, max_depth=45)
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

        ## Train algorithm with train set
        clf_svm = svm_cl.fit(X_tr,y_tr)
        clf_rf = rf_cl.fit(X_tr,y_tr)
        clf_knn = knn_cl.fit(X_tr,y_tr)
    print("Train data: {}".format(X_tr.shape[0]))
    print("Test data: {}".format(X_test.shape[0]))
    results_custom(clf_rf, clf_svm, clf_knn, y_test, labels, feats, "Face", X_test)


    #####################################################################

    print('Run_4: All data included...')
    X_tr, X_test, y_tr, y_test, labels, feats = dataset_preprocess("All_features", baseline)

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
    results_custom(clf_rf, clf_svm, clf_knn, y_test, labels, feats, "Fusion", X_test)
