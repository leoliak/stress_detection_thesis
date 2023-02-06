# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 23:14:06 2019

@author: leoni
"""

import numpy as np
# Written by Greg Ver Steeg (http://www.isi.edu/~gregv/npeet.html)
import scipy.spatial as ss
from scipy.special import digamma
from math import log
import numpy.random as nr
import random
import pandas as pd


# continuous estimators

def entropy(x, k=3, base=2):
    """
    The classic K-L k-nearest neighbor continuous entropy estimator x should be a list of vectors,
    e.g. x = [[1.3],[3.7],[5.1],[2.4]] if x is a one-dimensional scalar and we have four samples
    """

    assert k <= len(x)-1, "Set k smaller than num. samples - 1"
    d = len(x[0])
    N = len(x)
    intens = 1e-10  # small noise to break degeneracy, see doc.
    x = [list(p + intens * nr.rand(len(x[0]))) for p in x]
    tree = ss.cKDTree(x)
    nn = [tree.query(point, k+1, p=float('inf'))[0][k] for point in x]
    const = digamma(N)-digamma(k) + d*log(2)
    return (const + d*np.mean(map(log, nn)))/log(base)


def mi(x, y, k=3, base=2):
    """
    Mutual information of x and y; x, y should be a list of vectors, e.g. x = [[1.3],[3.7],[5.1],[2.4]]
    if x is a one-dimensional scalar and we have four samples
    """

    assert len(x) == len(y), "Lists should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    intens = 1e-10  # small noise to break degeneracy, see doc.
    x = [list(p + intens * nr.rand(len(x[0]))) for p in x]
    y = [list(p + intens * nr.rand(len(y[0]))) for p in y]
    points = zip2(x, y)
    # Find nearest neighbors in joint space, p=inf means max-norm
    tree = ss.cKDTree(points)
    dvec = [tree.query(point, k+1, p=float('inf'))[0][k] for point in points]
    a, b, c, d = avgdigamma(x, dvec), avgdigamma(y, dvec), digamma(k), digamma(len(x))
    return (-a-b+c+d)/log(base)


def cmi(x, y, z, k=3, base=2):
    """
    Mutual information of x and y, conditioned on z; x, y, z should be a list of vectors, e.g. x = [[1.3],[3.7],[5.1],[2.4]]
    if x is a one-dimensional scalar and we have four samples
    """

    assert len(x) == len(y), "Lists should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    intens = 1e-10  # small noise to break degeneracy, see doc.
    x = [list(p + intens * nr.rand(len(x[0]))) for p in x]
    y = [list(p + intens * nr.rand(len(y[0]))) for p in y]
    z = [list(p + intens * nr.rand(len(z[0]))) for p in z]
    points = zip2(x, y, z)
    # Find nearest neighbors in joint space, p=inf means max-norm
    tree = ss.cKDTree(points)
    dvec = [tree.query(point, k+1, p=float('inf'))[0][k] for point in points]
    a, b, c, d = avgdigamma(zip2(x, z), dvec), avgdigamma(zip2(y, z), dvec), avgdigamma(z, dvec), digamma(k)
    return (-a-b+c+d)/log(base)


def kldiv(x, xp, k=3, base=2):
    """
    KL Divergence between p and q for x~p(x), xp~q(x); x, xp should be a list of vectors, e.g. x = [[1.3],[3.7],[5.1],[2.4]]
    if x is a one-dimensional scalar and we have four samples
    """

    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    assert k <= len(xp) - 1, "Set k smaller than num. samples - 1"
    assert len(x[0]) == len(xp[0]), "Two distributions must have same dim."
    d = len(x[0])
    n = len(x)
    m = len(xp)
    const = log(m) - log(n-1)
    tree = ss.cKDTree(x)
    treep = ss.cKDTree(xp)
    nn = [tree.query(point, k+1, p=float('inf'))[0][k] for point in x]
    nnp = [treep.query(point, k, p=float('inf'))[0][k-1] for point in x]
    return (const + d*np.mean(map(log, nnp))-d*np.mean(map(log, nn)))/log(base)


# Discrete estimators
def entropyd(sx, base=2):
    """
    Discrete entropy estimator given a list of samples which can be any hashable object
    """

    return entropyfromprobs(hist(sx), base=base)


def midd(x, y):
    """
    Discrete mutual information estimator given a list of samples which can be any hashable object
    """

    return -entropyd(list(zip(x, y)))+entropyd(x)+entropyd(y)


def cmidd(x, y, z):
    """
    Discrete mutual information estimator given a list of samples which can be any hashable object
    """

    return entropyd(list(zip(y, z)))+entropyd(list(zip(x, z)))-entropyd(list(zip(x, y, z)))-entropyd(z)


def hist(sx):
    # Histogram from list of samples
    d = dict()
    for s in sx:
        d[s] = d.get(s, 0) + 1
    return map(lambda z: float(z)/len(sx), d.values())


def entropyfromprobs(probs, base=2):
    # Turn a normalized list of probabilities of discrete outcomes into entropy (base 2)
    return -sum(map(elog, probs))/log(base)


def elog(x):
    # for entropy, 0 log 0 = 0. but we get an error for putting log 0
    if x <= 0. or x >= 1.:
        return 0
    else:
        return x*log(x)


# Mixed estimators
def micd(x, y, k=3, base=2, warning=True):
    """ If x is continuous and y is discrete, compute mutual information
    """

    overallentropy = entropy(x, k, base)
    n = len(y)
    word_dict = dict()
    for sample in y:
        word_dict[sample] = word_dict.get(sample, 0) + 1./n
    yvals = list(set(word_dict.keys()))

    mi = overallentropy
    for yval in yvals:
        xgiveny = [x[i] for i in range(n) if y[i] == yval]
        if k <= len(xgiveny) - 1:
            mi -= word_dict[yval]*entropy(xgiveny, k, base)
        else:
            if warning:
                print("Warning, after conditioning, on y={0} insufficient data. Assuming maximal entropy in this case.".format(yval))
            mi -= word_dict[yval]*overallentropy
    return mi  # units already applied


# Utility functions
def vectorize(scalarlist):
    """
    Turn a list of scalars into a list of one-d vectors
    """

    return [(x,) for x in scalarlist]


def shuffle_test(measure, x, y, z=False, ns=200, ci=0.95, **kwargs):
    """
    Shuffle test
    Repeatedly shuffle the x-values and then estimate measure(x,y,[z]).
    Returns the mean and conf. interval ('ci=0.95' default) over 'ns' runs, 'measure' could me mi,cmi,
    e.g. Keyword arguments can be passed. Mutual information and CMI should have a mean near zero.
    """

    xp = x[:]   # A copy that we can shuffle
    outputs = []
    for i in range(ns):
        random.shuffle(xp)
        if z:
            outputs.append(measure(xp, y, z, **kwargs))
        else:
            outputs.append(measure(xp, y, **kwargs))
    outputs.sort()
    return np.mean(outputs), (outputs[int((1.-ci)/2*ns)], outputs[int((1.+ci)/2*ns)])


# Internal functions
def avgdigamma(points, dvec):
    # This part finds number of neighbors in some radius in the marginal space
    # returns expectation value of <psi(nx)>
    N = len(points)
    tree = ss.cKDTree(points)
    avg = 0.
    for i in range(N):
        dist = dvec[i]
        # subtlety, we don't include the boundary point,
        # but we are implicitly adding 1 to kraskov def bc center point is included
        num_points = len(tree.query_ball_point(points[i], dist-1e-15, p=float('inf')))
        avg += digamma(num_points)/N
    return avg


def zip2(*args):
    # zip2(x,y) takes the lists of vectors and makes it a list of vectors in a joint space
    # E.g. zip2([[1],[2],[3]],[[4],[5],[6]]) = [[1,4],[2,5],[3,6]]
    return [sum(sublist, []) for sublist in zip(*args)]

def information_gain(f1, f2):
    """
    This function calculates the information gain, where ig(f1,f2) = H(f1) - H(f1|f2)
    Input
    -----
    f1: {numpy array}, shape (n_samples,)
    f2: {numpy array}, shape (n_samples,)
    Output
    ------
    ig: {float}
    """

    ig = entropyd(f1) - conditional_entropy(f1, f2)
    return ig


def conditional_entropy(f1, f2):
    """
    This function calculates the conditional entropy, where ce = H(f1) - I(f1;f2)
    Input
    -----
    f1: {numpy array}, shape (n_samples,)
    f2: {numpy array}, shape (n_samples,)
    Output
    ------
    ce: {float}
        ce is conditional entropy of f1 and f2
    """

    ce = entropyd(f1) - midd(f1, f2)
    return ce


def su_calculation(f1, f2):
    """
    This function calculates the symmetrical uncertainty, where su(f1,f2) = 2*IG(f1,f2)/(H(f1)+H(f2))
    Input
    -----
    f1: {numpy array}, shape (n_samples,)
    f2: {numpy array}, shape (n_samples,)
    Output
    ------
    su: {float}
        su is the symmetrical uncertainty of f1 and f2
    """

    # calculate information gain of f1 and f2, t1 = ig(f1,f2)
    t1 = information_gain(f1, f2)
    # calculate entropy of f1, t2 = H(f1)
    t2 = entropyd(f1)
    # calculate entropy of f2, t3 = H(f2)
    t3 = entropyd(f2)
    # su(f1,f2) = 2*t1/(t2+t3)
    su = 2.0*t1/(t2+t3)
    return su

def merit_calculation(X, y):
    """
    This function calculates the merit of X given class labels y, where
    merits = (k * rcf)/sqrt(k+k*(k-1)*rff)
    rcf = (1/k)*sum(su(fi,y)) for all fi in X
    rff = (1/(k*(k-1)))*sum(su(fi,fj)) for all fi and fj in X
    Input
    ----------
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array}, shape (n_samples,)
        input class labels
    Output
    ----------
    merits: {float}
        merit of a feature subset X
    """

    n_samples, n_features = X.shape
    rff = 0
    rcf = 0
    for i in range(n_features):
        fi = X[:, i]
        rcf += su_calculation(fi, y)
        for j in range(n_features):
            if j > i:
                fj = X[:, j]
                rff += su_calculation(fi, fj)
    rff *= 2
    merits = rcf / np.sqrt(n_features + rff)
    return merits


def cfs(X, y):
    """
    This function uses a correlation based heuristic to evaluate the worth of features which is called CFS
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array}, shape (n_samples,)
        input class labels
    Output
    ------
    F: {numpy array}
        index of selected features
    Reference
    ---------
    Zhao, Zheng et al. "Advancing Feature Selection Research - ASU Feature Selection Repository" 2010.
    """

    n_samples, n_features = X.shape
    F = []
    # M stores the merit values
    M = []
    while True:
        merit = -100000000000
        idx = -1
        for i in range(n_features):
            if i not in F:
                F.append(i)
                # calculate the merit of current selected features
                t = merit_calculation(X[:, F], y)
                if t > merit:
                    merit = t
                    idx = i
                F.pop()
        F.append(idx)
        M.append(merit)
        if len(M) > 5:
            if M[len(M)-1] <= M[len(M)-2]:
                if M[len(M)-2] <= M[len(M)-3]:
                    if M[len(M)-3] <= M[len(M)-4]:
                        if M[len(M)-4] <= M[len(M)-5]:
                            break
    return np.array(F)


df_time = pd.read_csv('D:\Codes\\data\\full_feature_dataset.csv')

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
#st_scaler.fit(df)
#data = st_scaler.transform(df)
data = df.values

ar = cfs(data, labels)
feat_names = []
for i in range(0,len(ar)):
    feat_names.append(feats[ar[i]])
df2 = df[df.columns[ar]]

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

#x = sys.stdout
#x = Logger()
## Start of execution on output.txt file
        
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

df_time = df2
df_time['Condition'] = labels
voc = {"Condition": {"N":1, "I":2, "T":2, "R":1}}
df_t = df_time
df_t.replace(voc, inplace=True)
#df2 = pd.read_csv('time_feat_data.csv')
#df2 = df2.loc[:, ~df2.columns.str.match('Unnamed')]
labels = df_t.Condition
df_2 = df_t.drop(df_t.columns[[12]],axis=1)
feats = df.columns
st_scaler.fit(df_2)
data = st_scaler.transform(df_2)

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