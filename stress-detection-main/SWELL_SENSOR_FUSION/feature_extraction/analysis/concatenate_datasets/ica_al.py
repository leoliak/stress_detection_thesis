import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(rc={'figure.figsize':(11.7,8.27)})

def g(x):
    return np.tanh(x)
def g_der(x):
    return 1 - g(x) * g(x)

def center(X):
    X = np.array(X)
    mean = X.mean()
    return X - mean

def whitening(X):
    cov = np.cov(X)
    d, E = np.linalg.eigh(cov)
    D = np.diag(d)
    D_inv = np.sqrt(np.linalg.inv(D))
    X_whiten = np.dot(E, np.dot(D_inv, np.dot(E.T, X)))
    return X_whiten

def calculate_new_w(w, X):
    w_new = (X * g(np.dot(w.T, X))).mean(axis=1) - g_der(np.dot(w.T, X)).mean() * w
    w_new /= np.sqrt((w_new ** 2).sum())
    return w_new

def ica(X, iterations, tolerance=1e-5):
    X = center(X)
    X = whitening(X)
    components_nr = X.shape[0]
    W = np.zeros((components_nr, components_nr), dtype=X.dtype)
    for i in range(components_nr):       
        w = np.random.rand(components_nr)       
        for j in range(iterations):
            print(i)
            w_new = calculate_new_w(w, X)           
            if i >= 1:
                w_new -= np.dot(np.dot(w_new, W[:i].T), W[:i])           
            distance = np.abs(np.abs((w * w_new).sum()) - 1)           
            w = w_new           
            if distance < tolerance:
                break               
        W[i, :] = w       
    S = np.dot(W, X)   
    return S

def plot_mixture_sources_predictions(X, sig, S):
    fig = plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(sig)
    plt.title("Source")
    plt.subplot(3, 1, 2)
    for x in X:
        plt.plot(x)
    plt.title("Mixtures")
    plt.subplot(3, 1, 3)
    for s in S:
        plt.plot(s)
    plt.title("predicted sources")
    fig.tight_layout()
    plt.show()
    return

df = pd.read_csv('time_domain_signals_new_time/data_pp1_18-9-2012_c1.csv')
ecg = df.heart
rang = 10000
sig = ecg[0:rang]

sig = np.array(sig)/10
#s1 = np.zeros((rang,2))
#x = sig.reshape((rang,1))
#s1[:,0] = sig
#x = x.T
#s1[:,1] = x
#A = np.array(([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]))

s1 = sig.reshape((rang,1))
A = np.array(([[1.0], [1.0], [1.0]]))
X = np.dot(s1, A.T)
#X = np.array(sig)
X = X.T
s1 = s1
plt.plot(s1)
S = ica(s1, iterations=1000)
plot_mixture_sources_predictions(X, sig, S)



#plt.figure()
from sklearn.decomposition import FastICA


icaa = FastICA(n_components=3)
S_ = icaa.fit_transform(s1)
fig = plt.figure()
models = [s1, S_]
names = ['mixtures', 'real sources', 'predicted sources']
models = [s1, S_]
names = ['real sources', 'predicted sources']
colors = ['red', 'blue', 'orange']
for i, (name, model) in enumerate(zip(names, models)):
    plt.subplot(4, 1, i+1)
    plt.title(name)
    for sig, color in zip (model.T, colors):
        plt.plot(sig, color=color)
fig.tight_layout()        
plt.show()

