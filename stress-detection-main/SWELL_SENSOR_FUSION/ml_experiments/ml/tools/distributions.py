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
baseline = False
trainset_eval = False

list_nonbaseline = ['datasets/facereader_data_new.csv', 'datasets/kinect_data_new.csv', 'datasets/physio_data_new.csv', 'datasets/full_feature_dataset.csv']
list_baseline = ['baseline_datasets/face_baseline_data.csv', 'baseline_datasets/kinect_baseline_data.csv', 'baseline_datasets/physio_baselinne_data.csv', 'baseline_datasets/full_feature_baseline_dataset.csv']
dict_datasets = {"Face" : 0, "Kinect" : 1, "Physio" : 2, "All_features" : 3}


#####################################################################
#####################################################################
#####################################################################
    
    
def rmv_non_important_features():
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

    

def dataset_preprocess(id_name, baseline):
    print("Load and preprocess %s data.." %id_name)
    df_name = dict_datasets[id_name]
    if not baseline:
        df_time = pd.read_csv(list_nonbaseline[df_name])
        voc = {"Condition": {"N":0, "I":1, "T":1, "R":0}}
    else:
        df_time = pd.read_csv(list_baseline[df_name])
        voc = {"Condition": {"N":0, "S":1}}
    st_scaler = preprocessing.StandardScaler()
    df_t = df_time.copy()
    df_t.replace(voc, inplace=True)
    labels = df_t.Condition
    df = df_t.drop(df_t.columns[[0,1,2,3,4]],axis=1)
    #df = df.drop(['SD1','SD2','SDSD'],axis=1)
    df = df.fillna(df.mean())
    df = df.replace([-np.inf], 0.0)
    feats = df.columns
    st_scaler.fit(df)
    data = st_scaler.transform(df)   
    X_tr, X_test, y_tr, y_test = train_test_split(data, labels, test_size = 0.3, random_state = 42)
    return df_t, df, [X_tr, X_test, y_tr, y_test]
        

def boxplt(data):
    fig = plt.figure()
    plt.boxplot(data)
    plt.show()
    
    

#####################################################################
#####################################################################
#####################################################################



print('Run_1: Facereader data...')
df_t, df, data = dataset_preprocess("Physio", baseline)

dd0 = df[df_t.Condition == 0]
dd1 = df[df_t.Condition == 1]

# boxplt(dd0.RMSSD)
feats = dd1.columns.values.tolist()

import seaborn as sns
# Draw the density plot
for feat in feats:
    sns.distplot(dd0[feat], hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = 0)
    sns.distplot(dd1[feat], hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = 1)
        
    # Plot formatting
    fig = plt.figure()
    plt.legend(prop={'size': 16}, title = 'Stress')
    plt.title('Density Plot ' + feat)
    plt.xlabel(feat)
    plt.ylabel('Density')

