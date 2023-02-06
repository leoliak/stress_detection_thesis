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


st_scaler = preprocessing.StandardScaler()

#sdf_time = pd.read_csv('full_dataset.csv')
#
#voc = {"Condition": {"N":1, "I":2, "T":2, "R":1}}
#df_t = df_time
#df_t.replace(voc, inplace=True)
#
#df_skin = df_t[['Skin_mean','Skin_dev','Skin_var','Skin_max','Skin_min']]
#labels = df_t.Condition
#df = df_t.drop(df_t.columns[[0,1,2,3,4]],axis=1)
##df = df.drop(['SD1','SD2','SDSD'],axis=1)
##df = df.drop(['Skin_mean','Skin_dev','Skin_var','Skin_max','Skin_min'],axis=1)
#
#print('Run_1: No skin data...')
#df = df.fillna(df.mean())
#df = df.replace([-np.inf], 0.0)
#
#pca = PCA(n_components=10)
#pca.fit(df)
#
#
#feats = df.columns
#st_scaler.fit(df)
#data = st_scaler.transform(df)
#
#
#X_tr, X_test, y_tr, y_test = train_test_split(data, labels, test_size = 0.3, random_state = 42)
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

#df_time = pd.read_csv('full_dataset.csv')
#
##df2 = pd.read_csv('time_feat_data.csv')
##df2 = df2.loc[:, ~df2.columns.str.match('Unnamed')]
#
#voc = {"Condition": {"N":1, "I":2, "T":2, "R":1}}
#df_t = df_time
#df_t.replace(voc, inplace=True)
#
#df_skin = df_t[['Skin_mean','Skin_dev','Skin_var','Skin_max','Skin_min']]
#labels = df_t.Condition
#df = df_t.drop(df_t.columns[[0,1,2,3,4]],axis=1)
##df = df.drop(['SD1','SD2','SDSD'],axis=1)
##df = df.drop(['Skin_mean','Skin_dev','Skin_var','Skin_max','Skin_min'],axis=1)
#print('Run_2: With skin data...')
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
st_scaler = preprocessing.StandardScaler()
df_time = pd.read_csv('D:\Codes\\data\\full_feature_dataset.csv')

voc = {"Condition": {"N":1, "I":2, "T":2, "R":1}}
df_t = df_time
df_t.replace(voc, inplace=True)

df_skin = df_t[['Skin_mean','Skin_dev','Skin_var','Skin_max','Skin_min']]
labels = df_t.Condition
df = df_t.drop(df_t.columns[[0,1,2,3,4]],axis=1)
#df = df.drop(['SD1','SD2','SDSD'],axis=1)
#df = df.drop(['Skin_mean','Skin_dev','Skin_var','Skin_max','Skin_min'],axis=1)
print('Run_3: All data included...')

df = df.fillna(df.mean())
df = df.replace([-np.inf], 0.0)

feats = df.columns
st_scaler.fit(df)
data = st_scaler.transform(df)

X_tr, X_test, y_tr, y_test = train_test_split(data, labels, test_size = 0.3, random_state = 42)


## Classifiers initialization
svm_cl = SVC(kernel='rbf', C=2.91, gamma=0.01, probability=True)
rf_cl = RandomForestClassifier(n_estimators=400, max_depth=25)
knn_cl = KNeighborsClassifier(n_neighbors = 3)

## 10-fold train set
scores_svm = cross_val_score(svm_cl, X_tr, y_tr, cv=10)
scores_rf = cross_val_score(rf_cl, X_tr, y_tr, cv=10)
scores_knn = cross_val_score(knn_cl, X_tr, y_tr, cv=10)

print('10-fold Score for SVM:', scores_svm.mean())
print('10-fold Score for RF:',scores_rf.mean())
print('10-fold Score for KNN:',scores_knn.mean())

## Train algorithm with train set
clf_svm = svm_cl.fit(X_tr,y_tr)
clf_rf = rf_cl.fit(X_tr,y_tr)
clf_knn = knn_cl.fit(X_tr,y_tr)

## Prediction class of test set
y_pred_svm = clf_svm.predict(X_test)
y_pred_rf = clf_rf.predict(X_test)
y_pred_knn = clf_knn.predict(X_test)

## Prediction probability of test set
y_pred_svm_prob = clf_svm.predict_proba(X_test)
y_pred_rf_prob = clf_rf.predict_proba(X_test)
y_pred_knn_prob = clf_knn.predict_proba(X_test)

## Print results for test set
a_svm = metrics.accuracy_score(y_test, y_pred_svm)
a_rf = metrics.accuracy_score(y_test, y_pred_rf)
a_knn = metrics.accuracy_score(y_test, y_pred_knn)

print('\n')
print('Test Score for SVM:', a_svm)
print('Test Score for RF:',a_rf)
print('Test Score for KNN:',a_knn)

## Calculate Comfusion Matrix
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

## Count each class appearance in test and whole dataset
unique, counts = np.unique(y_test, return_counts=True)
dic = dict(zip(unique, counts))
unique, counts = np.unique(labels, return_counts=True)
dic2 = dict(zip(unique, counts))

## Keep RF feature importances and keep the best
feat_importance = clf_rf.feature_importances_
dictionary = dict(zip(feats, feat_importance))
x = dictionary
sorted_x = sorted(x.items(), key=lambda kv: kv[1],reverse=True)
li = []
for (key, value) in x.items():
   # Check if key is even then add pair to new dictionary
   if value<=0.004:
       li.append(key)


#df_time = pd.read_csv('D:\Codes\\data\\full_feature_dataset.csv')
#
##df2 = pd.read_csv('time_feat_data.csv')
##df2 = df2.loc[:, ~df2.columns.str.match('Unnamed')]
#
#voc = {"Condition": {"N":1, "I":2, "T":2, "R":1}}
#df_t = df_time
#df_t.replace(voc, inplace=True)
#
#df_skin = df_t[['Skin_mean','Skin_dev','Skin_var','Skin_max','Skin_min']]
#labels = df_t.Condition
#df = df_t.drop(df_t.columns[[0,1,2,3,4]],axis=1)
#df = df.drop(li,axis=1)
##df = df.drop(['SD1','SD2','SDSD'],axis=1)
##df = df.drop(['Skin_mean','Skin_dev','Skin_var','Skin_max','Skin_min'],axis=1)
#print('Run_3: All data included...')
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
#svm_cl = SVC(kernel='rbf', C=2.91, gamma=0.01, probability=True)
#rf_cl = RandomForestClassifier(n_estimators=400, max_depth=25)
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
#print('\n')
#print('RF Comfusion Matrix:')
#print_cm(c_m_rf,['Relax','Stress'])
#print('\n')
#print('KNN Comfusion Matrix:')
#print_cm(c_m_knn,['Relax','Stress'])
















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
##
##df2.replace(voc, inplace=True)
##labels2 = df2.Condition
##df21 = df2.drop(df2.columns[[0,1,2,3,4]],axis=1)
##st_scaler.fit(df21)
##data2 = st_scaler.transform(df21)
##X_tr2, X_test2, y_tr2, y_test2 = train_test_split(data2, labels2, test_size = 0.3, random_state = 42)
##
##svm_cl2 = SVC(kernel='rbf', C=2.61, gamma=0.1, probability=True)
##scores2 = cross_val_score(svm_cl2, X_tr2, y_tr2, cv=10)
##print(scores2.mean())
##
##clf2 = svm_cl2.fit(X_tr2,y_tr2)
##y_pred2 = clf2.predict(X_test2)
##
##c_m2 = metrics.confusion_matrix(y_test2, y_pred2)