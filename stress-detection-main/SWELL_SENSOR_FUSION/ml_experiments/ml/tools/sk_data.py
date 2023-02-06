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
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2, RFE

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
                self.models={
                             'Random_Forest':[RandomForestClassifier(),dict(n_estimators = np.arange(200,1300,100),max_depth = np.linspace(20, 50, 7, endpoint=True))]}
            else:
              self.models={'SVM':[SVC(probability=True),dict(kernel=['rbf'],gamma=np.logspace(-3, 1, 5),C=np.arange(0.01, 3.01, 0.1))],\
                             'LinearSVM':[LinearSVC(dual=False),dict(penalty=['l1','l2'],tol=np.logspace(-8, -2, 7),C=np.arange(0.01, 3.01, 0.2))],\
                             'LogisticRegression':[lr(solver='lbfgs',multi_class='auto'),dict(C=np.arange(0.1,3,0.1))],\
                             'KNN':[KNeighborsClassifier(),dict(n_neighbors=np.arange(1, 100))],\
                      
                      
                      
                      'SVM':[SVC(kernel='rbf',probability=True),dict(gamma=np.logspace(-9, 1, 11),C=np.arange(0.01, 3.01, 0.2))],\
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



st_scaler = preprocessing.StandardScaler()

#df_time = pd.read_csv('datasets/facereader_data_new.csv')
#
#voc = {"Condition": {"N":1, "I":2, "T":2, "R":1}}
#df_t = df_time
#df_t.replace(voc, inplace=True)
#
#df_time = pd.read_csv('baseline_datasets/face_baseline_data.csv')
#voc = {"Condition": {"N":1, "S":2}}
#df_t = df_time
#df_t.replace(voc, inplace=True)
#
#
#labels = df_t.Condition
#df = df_t.drop(df_t.columns[[0,1,2,3,4]],axis=1)
##df = df.drop(['SD1','SD2','SDSD'],axis=1)
##df = df.drop(['Skin_mean','Skin_dev','Skin_var','Skin_max','Skin_min'],axis=1)
#
#print('Run_1: Facereader data...')
#df = df.fillna(df.mean())
#df = df.replace([-np.inf], 0.0)
#
#
#feats = df.columns
#st_scaler.fit(df)
#data = st_scaler.transform(df)
#
#X_tr, X_test, y_tr, y_test = train_test_split(data, labels, test_size = 0.3, random_state = 42)
#
#print('Start training phase..')
#classifiers = Classifiers(X_tr, y_tr,True)
#models = classifiers.models
#
#svm_cl = SVC(kernel='rbf', C=2.41, gamma=0.1, probability=True)
#rf_cl = RandomForestClassifier(n_estimators=300, max_depth=45)
#knn_cl = KNeighborsClassifier(n_neighbors = 1)
#scores_svm = cross_val_score(svm_cl, X_tr, y_tr, cv=10)
#scores_rf = cross_val_score(rf_cl, X_tr, y_tr, cv=10)
#scores_knn = cross_val_score(knn_cl, X_tr, y_tr, cv=10)
#
#
#print('10-fold Score for SVM:', scores_svm.mean())
#print('10-fold Score for RF:',scores_rf.mean())
#print('10-fold Score for KNN:',scores_knn.mean())
#
#clf_svm = svm_cl.fit(X_tr,y_tr)
#clf_rf = rf_cl.fit(X_tr,y_tr)
#clf_knn = knn_cl.fit(X_tr,y_tr)
#
#feat_importance = clf_rf.feature_importances_
#dictionary = dict(zip(feats, feat_importance))
#
#y_pred_svm = clf_svm.predict(X_test)
#y_pred_rf = clf_rf.predict(X_test)
#y_pred_knn = clf_knn.predict(X_test)
#
#a_svm = metrics.accuracy_score(y_test, y_pred_svm)
#a_rf = metrics.accuracy_score(y_test, y_pred_rf)
#a_knn = metrics.accuracy_score(y_test, y_pred_knn)
#
#print('\n')
#print('Test Score for SVM:', a_svm)
#print('Test Score for RF:',a_rf)
#print('Test Score for KNN:',a_knn)
#c_m_svm = metrics.confusion_matrix(y_test, y_pred_svm)
#c_m_rf = metrics.confusion_matrix(y_test, y_pred_rf)
#c_m_knn = metrics.confusion_matrix(y_test, y_pred_knn)
#print('\n')
#print('SVM Comfusion Matrix:')
#print_cm(c_m_svm,['Relax','Stress'])
#print('RF Comfusion Matrix:')
#print_cm(c_m_rf,['Relax','Stress'])
#print('KNN Comfusion Matrix:')
#print_cm(c_m_knn,['Relax','Stress'])
#
#unique, counts = np.unique(y_test, return_counts=True)
#dic = dict(zip(unique, counts))
#unique, counts = np.unique(labels, return_counts=True)
#dic2 = dict(zip(unique, counts))
#
#
#
#
#
#st_scaler = preprocessing.StandardScaler()
#
#df_time = pd.read_csv('datasets/kinect_data_new.csv')
##df2 = pd.read_csv('time_feat_data.csv')
##df2 = df2.loc[:, ~df2.columns.str.match('Unnamed')]
#
#voc = {"Condition": {"N":1, "I":2, "T":2, "R":1}}
#df_t = df_time
#df_t.replace(voc, inplace=True)
#
#df_time = pd.read_csv('baseline_datasets/kinect_baseline_data.csv')
#voc = {"Condition": {"N":1, "S":2}}
#df_t = df_time
#df_t.replace(voc, inplace=True)
#
#labels = df_t.Condition
#df = df_t.drop(df_t.columns[[0,1,2,3,4]],axis=1)
##df = df.drop(['SD1','SD2','SDSD'],axis=1)
##df = df.drop(['Skin_mean','Skin_dev','Skin_var','Skin_max','Skin_min'],axis=1)
#print('Run_2: Kinect data...')
#
#df = df.fillna(df.mean())
#df = df.replace([-np.inf], 0.0)
#
#feats = df.columns
#st_scaler.fit(df)
#data = st_scaler.transform(df)
#
#X_tr, X_test, y_tr, y_test = train_test_split(data, labels, test_size = 0.3, random_state = 42)
#
#print('Start training phase..')
#classifiers = Classifiers(X_tr, y_tr,True)
#models = classifiers.models
#
#svm_cl = SVC(kernel='rbf', C=2.41, gamma=0.1, probability=True)
#rf_cl = RandomForestClassifier(n_estimators=300, max_depth=45)
#knn_cl = KNeighborsClassifier(n_neighbors = 1)
#scores_svm = cross_val_score(svm_cl, X_tr, y_tr, cv=10)
#scores_rf = cross_val_score(rf_cl, X_tr, y_tr, cv=10)
#scores_knn = cross_val_score(knn_cl, X_tr, y_tr, cv=10)
#
#
#print('10-fold Score for SVM:', scores_svm.mean())
#print('10-fold Score for RF:',scores_rf.mean())
#print('10-fold Score for KNN:',scores_knn.mean())
#
#clf_svm = svm_cl.fit(X_tr,y_tr)
#clf_rf = rf_cl.fit(X_tr,y_tr)
#clf_knn = knn_cl.fit(X_tr,y_tr)
#
#feat_importance = clf_rf.feature_importances_
#dictionary = dict(zip(feats, feat_importance))
#
#y_pred_svm = clf_svm.predict(X_test)
#y_pred_rf = clf_rf.predict(X_test)
#y_pred_knn = clf_knn.predict(X_test)
#
#a_svm = metrics.accuracy_score(y_test, y_pred_svm)
#a_rf = metrics.accuracy_score(y_test, y_pred_rf)
#a_knn = metrics.accuracy_score(y_test, y_pred_knn)
#
#print('\n')
#print('Test Score for SVM:', a_svm)
#print('Test Score for RF:',a_rf)
#print('Test Score for KNN:',a_knn)
#c_m_svm = metrics.confusion_matrix(y_test, y_pred_svm)
#c_m_rf = metrics.confusion_matrix(y_test, y_pred_rf)
#c_m_knn = metrics.confusion_matrix(y_test, y_pred_knn)
#print('\n')
#print('SVM Comfusion Matrix:')
#print_cm(c_m_svm,['Relax','Stress'])
#print('RF Comfusion Matrix:')
#print_cm(c_m_rf,['Relax','Stress'])
#print('KNN Comfusion Matrix:')
#print_cm(c_m_knn,['Relax','Stress'])
#
#unique, counts = np.unique(y_test, return_counts=True)
#dic = dict(zip(unique, counts))
#unique, counts = np.unique(labels, return_counts=True)
#dic2 = dict(zip(unique, counts))




## Data load and preprocessing
#st_scaler = preprocessing.StandardScaler()
#
#df_time = pd.read_csv('datasets/physio_data_new.csv')
#voc = {"Condition": {"N":1, "I":2, "T":2, "R":1}}
#df_t = df_time
#df_t.replace(voc, inplace=True)
#
#df_time = pd.read_csv('baseline_datasets/physio_baseline_data.csv')
#voc = {"Condition": {"N":1, "S":2}}
#df_t = df_time
#df_t.replace(voc, inplace=True)
#
#df_skin = df_t[['Skin_mean','Skin_dev','Skin_var','Skin_max','Skin_min']]
#labels = df_t.Condition
#df = df_t.drop(df_t.columns[[0,1,2,3,4]],axis=1)
##df = df.drop(['SD1','SD2','SDSD'],axis=1)
##df = df.drop(['Skin_mean','Skin_dev','Skin_var','Skin_max','Skin_min'],axis=1)
#print('Run_3: Physio data...')
#
#df = df.fillna(df.mean())
#df = df.replace([-np.inf], 0.0)
#df = df.drop(['ln_vlf'], axis=1)
#feats = df.columns
#st_scaler.fit(df)
#data = st_scaler.transform(df)
#
#X_tr, X_test, y_tr, y_test = train_test_split(data, labels, test_size = 0.3, random_state = 42)
#
#print('Start training phase..')
#classifiers = Classifiers(X_tr, y_tr,True)
#models = classifiers.models
#
### Classifiers initialization
#svm_cl = SVC(kernel='rbf', C=2.05, gamma=0.1, probability=True)
#rf_cl = RandomForestClassifier(n_estimators=400, max_depth=25)
#knn_cl = KNeighborsClassifier(n_neighbors = 3)
#
### 10-fold train set
#scores_svm = cross_val_score(svm_cl, X_tr, y_tr, cv=10)
#scores_rf = cross_val_score(rf_cl, X_tr, y_tr, cv=10)
#scores_knn = cross_val_score(knn_cl, X_tr, y_tr, cv=10)
#
#print('10-fold Score for SVM:', scores_svm.mean())
#print('10-fold Score for RF:',scores_rf.mean())
#print('10-fold Score for KNN:',scores_knn.mean())
#
### Train algorithm with train set
#clf_svm = svm_cl.fit(X_tr,y_tr)
#clf_rf = rf_cl.fit(X_tr,y_tr)
#clf_knn = knn_cl.fit(X_tr,y_tr)
#
### Prediction class of test set
#y_pred_svm = clf_svm.predict(X_test)
#y_pred_rf = clf_rf.predict(X_test)
#y_pred_knn = clf_knn.predict(X_test)
#
### Prediction probability of test set
#y_pred_svm_prob = clf_svm.predict_proba(X_test)
#y_pred_rf_prob = clf_rf.predict_proba(X_test)
#y_pred_knn_prob = clf_knn.predict_proba(X_test)
#
### Print results for test set
#a_svm = metrics.accuracy_score(y_test, y_pred_svm)
#a_rf = metrics.accuracy_score(y_test, y_pred_rf)
#a_knn = metrics.accuracy_score(y_test, y_pred_knn)
#
#print('\n')
#print('Test Score for SVM:', a_svm)
#print('Test Score for RF:',a_rf)
#print('Test Score for KNN:',a_knn)
#
### Calculate Comfusion Matrix
#c_m_svm = metrics.confusion_matrix(y_test, y_pred_svm)
#c_m_rf = metrics.confusion_matrix(y_test, y_pred_rf)
#c_m_knn = metrics.confusion_matrix(y_test, y_pred_knn)
#
#print('\n')
#print('SVM Comfusion Matrix:')
#print_cm(c_m_svm,['Relax','Stress'])
#print('\n')
#print('RF Comfusion Matrix:')
#print_cm(c_m_rf,['Relax','Stress'])
#print('\n')
#print('KNN Comfusion Matrix:')
#print_cm(c_m_knn,['Relax','Stress'])
#
### Count each class appearance in test and whole dataset
#unique, counts = np.unique(y_test, return_counts=True)
#dic = dict(zip(unique, counts))
#unique, counts = np.unique(labels, return_counts=True)
#dic2 = dict(zip(unique, counts))
#
### Keep RF feature importances and keep the best
#feat_importance = clf_rf.feature_importances_
#dictionary = dict(zip(feats, feat_importance))
#x = dictionary
#sorted_x = sorted(x.items(), key=lambda kv: kv[1],reverse=True)
#li = []
#for (key, value) in x.items():
#   # Check if key is even then add pair to new dictionary
#   if value<=0.004:
#       li.append(key)
#



st_scaler = preprocessing.StandardScaler()
df_time = pd.read_csv('datasets/full_feature_dataset.csv')

voc = {"Condition": {"N":1, "I":2, "T":2, "R":1}}
df_t = df_time
df_t.replace(voc, inplace=True)

df_time = pd.read_csv('baseline_datasets/full_feature_baseline_dataset.csv')
voc = {"Condition": {"N":1, "S":2}}
df_t = df_time
df_t.replace(voc, inplace=True)

df_skin = df_t[['Skin_mean','Skin_dev','Skin_var','Skin_max','Skin_min']]
labels = df_t.Condition
df = df_t.drop(df_t.columns[[0,1,2,3,4]],axis=1)
#df = df.drop(['SD1','SD2','SDSD'],axis=1)
#df = df.drop(['Skin_mean','Skin_dev','Skin_var','Skin_max','Skin_min'],axis=1)
print('Run_4: All data included...')

df = df.fillna(df.mean())
df = df.replace([-np.inf], 0.0)
df = df.drop(['ln_vlf'], axis=1)

feats = df.columns
st_scaler.fit(df)
data = st_scaler.transform(df)


X_tr, X_test, y_tr, y_test = train_test_split(data, labels, test_size = 0.3, random_state = 42)

print('Start training phase..')
classifiers = Classifiers(X_tr, y_tr,True)
models = classifiers.models

svm_cl = SVC(kernel='rbf', C=2.91, gamma=0.01, probability=True)
rf_cl = RandomForestClassifier(n_estimators=400, max_depth=25)
knn_cl = KNeighborsClassifier(n_neighbors = 1)
scores_svm = cross_val_score(svm_cl, X_tr, y_tr, cv=10)
scores_rf = cross_val_score(rf_cl, X_tr, y_tr, cv=10)
scores_knn = cross_val_score(knn_cl, X_tr, y_tr, cv=10)


print('10-fold Score for SVM:', scores_svm.mean())
print('10-fold Score for RF:',scores_rf.mean())
print('10-fold Score for KNN:',scores_knn.mean())

clf_svm = svm_cl.fit(X_tr,y_tr)
clf_rf = rf_cl.fit(X_tr,y_tr)
clf_knn = knn_cl.fit(X_tr,y_tr)

feat_importance = clf_rf.feature_importances_
dictionary = dict(zip(feats, feat_importance))

y_pred_svm = clf_svm.predict(X_test)
y_pred_rf = clf_rf.predict(X_test)
y_pred_knn = clf_knn.predict(X_test)

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


#
#df_time = pd.read_csv('physio_dataset.csv')
#df_time = df_time.drop(['timestamp.1','Condition.1'],axis=1)
##df2 = pd.read_csv('time_feat_data.csv')
##df2 = df2.loc[:, ~df2.columns.str.match('Unnamed')]
#
#voc = {"Condition": {"N":1, "I":2, "T":2, "R":1}}
#df_t = df_time
#df_t.replace(voc, inplace=True)
#
#df_skin = df_t[['Skin_mean','Skin_dev','Skin_var','Skin_max','Skin_min']]
#labels = df_t.Condition
#df = df_t.drop(df_t.columns[[0,1,2,3]],axis=1)
##df = df.drop(['SD1','SD2','SDSD'],axis=1)
##df = df.drop(['Skin_mean','Skin_dev','Skin_var','Skin_max','Skin_min'],axis=1)
#print('Run_4: Physio data included...')
#
#df = df.fillna(df.mean())
#df = df.replace([-np.inf], 0.0)
#
#feats = df.columns
#st_scaler.fit(df)
#data = st_scaler.transform(df)
#
#
#X_tr, X_test, y_tr, y_test = train_test_split(data, labels, test_size = 0.3, random_state = 42)
#
#svm_cl = SVC(kernel='rbf', C=2.11, gamma=0.1, probability=True)
#rf_cl = RandomForestClassifier(n_estimators=700, max_depth=35)
#knn_cl = KNeighborsClassifier(n_neighbors = 7)
#scores_svm = cross_val_score(svm_cl, X_tr, y_tr, cv=10)
#scores_rf = cross_val_score(rf_cl, X_tr, y_tr, cv=10)
#scores_knn = cross_val_score(knn_cl, X_tr, y_tr, cv=10)
#
#
#print('10-fold Score for SVM:', scores_svm.mean())
#print('10-fold Score for RF:',scores_rf.mean())
#print('10-fold Score for KNN:',scores_knn.mean())
#
#clf_svm = svm_cl.fit(X_tr,y_tr)
#clf_rf = rf_cl.fit(X_tr,y_tr)
#clf_knn = knn_cl.fit(X_tr,y_tr)
#
#feat_importance = clf_rf.feature_importances_
#dictionary = dict(zip(feats, feat_importance))
#
#y_pred_svm = clf_svm.predict(X_test)
#y_pred_rf = clf_rf.predict(X_test)
#y_pred_knn = clf_knn.predict(X_test)
#
#a_svm = metrics.accuracy_score(y_test, y_pred_svm)
#a_rf = metrics.accuracy_score(y_test, y_pred_rf)
#a_knn = metrics.accuracy_score(y_test, y_pred_knn)
#
#print('\n')
#print('Test Score for SVM:', a_svm)
#print('Test Score for RF:',a_rf)
#print('Test Score for KNN:',a_knn)
#c_m_svm = metrics.confusion_matrix(y_test, y_pred_svm)
#c_m_rf = metrics.confusion_matrix(y_test, y_pred_rf)
#c_m_knn = metrics.confusion_matrix(y_test, y_pred_knn)
#print('\n')
#print('SVM Comfusion Matrix:')
#print_cm(c_m_svm,['Relax','Stress'])
#print('\n')
#print('RF Comfusion Matrix:')
#print_cm(c_m_rf,['Relax','Stress'])
#print('\n')
#print('KNN Comfusion Matrix:')
#print_cm(c_m_knn,['Relax','Stress'])
#
#unique, counts = np.unique(y_test, return_counts=True)
#dic = dict(zip(unique, counts))
#unique, counts = np.unique(labels, return_counts=True)
#dic2 = dict(zip(unique, counts))
#
#df2.replace(voc, inplace=True)
#labels2 = df2.Condition
#df21 = df2.drop(df2.columns[[0,1,2,3,4]],axis=1)
#st_scaler.fit(df21)
#data2 = st_scaler.transform(df21)
#X_tr2, X_test2, y_tr2, y_test2 = train_test_split(data2, labels2, test_size = 0.3, random_state = 42)
#
#svm_cl2 = SVC(kernel='rbf', C=2.61, gamma=0.1, probability=True)
#scores2 = cross_val_score(svm_cl2, X_tr2, y_tr2, cv=10)
#print(scores2.mean())
#
#clf2 = svm_cl2.fit(X_tr2,y_tr2)
#y_pred2 = clf2.predict(X_test2)
#
#c_m2 = metrics.confusion_matrix(y_test2, y_pred2)