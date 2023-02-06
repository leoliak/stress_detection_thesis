# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 13:02:50 2019

@author: leoni
"""

import pandas as pd
import numpy as np
import math
import re
from os import listdir
from os.path import isfile, join

#d1 = pd.read_csv('D:\Codes\\data\\datasets\\data_1\\physio_data_new.csv')
#d2 = pd.read_csv('D:\Codes\data\datasets\data_1\\kinect_data_new.csv')
#d3 = pd.read_csv('D:\Codes\data\datasets\data_1\\facereader_data_new.csv')

d1 = pd.read_csv("D:\Codes\data\\unified_cases\\physio_baseline_data.csv")
d2 = pd.read_csv("D:\Codes\data\\unified_cases\\kinect_baseline_data.csv")
d3 = pd.read_csv("D:\Codes\data\\unified_cases\\face_baseline_data.csv")
#
d1 = d1.drop(['Minutes'],axis=1)
d2 = d2.drop(['Minutes'],axis=1)
d3 = d3.drop(['Minutes'],axis=1)

v1 = d1.timestamp.copy()
v2 = d2.timestamp.copy()
v3 = d3.timestamp.copy()

for i in range(0,v1.shape[0]):
    x = v1[i]
    x1 = x[9:]
    v1[i] = x1
for i in range(0,v2.shape[0]):
    x = v2[i]
    x1 = x[9:]
    v2[i]= x1
for i in range(0,v3.shape[0]):
    x = v3[i]
    x1 = x[9:]
    v3[i] = x1
d1.insert(4,'timestamp_merge',v1,True)
d2.insert(4,'timestamp_merge',v2,True)
d3.insert(4,'timestamp_merge',v3,True)

t = pd.merge(d2,d3,on=['PP','C','timestamp_merge'])
t = t.drop(['timestamp_y','Condition_y'],axis=1)
t = t.rename(columns={'timestamp_x': 'timestamp','Condition_x': 'Condition'})
t_1 = pd.merge(d1,t,on=['PP','C','timestamp_merge'])
t_1 = t_1.drop(['timestamp_y','Condition_y'],axis=1)
t_1 = t_1.rename(columns={'timestamp_x': 'timestamp','Condition_x': 'Condition'})

t_1 = t_1.replace([-np.inf], 0.0)
t_1.to_csv('D:\Codes\\data\\full_feature_baseline_dataset.csv',index=False)


#################################################################################################
#mypath = 'D:\Codes\\data\\physio\\whole'
#names = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#for pp in range(1,26):
#    d = []
#    dfs = []
#    for i in range(0,len(names)):
#        x = re.findall("\d+", names[i])
#        if x[0] == str(pp):
#            d.append('D:\Codes\data\\physio\\whole\\'+names[i])
#    if len(d)==0: continue
#    x = re.findall("\d+",d[0])
#    if len(d)==1:
#        d1 = pd.read_csv(d[0])
#        st = 'D:\Codes\data\\unified_cases\\physio\\pp_'+x[0]+'.csv'
#        df = d1
#        df.to_csv(st,index=False)
#    elif len(d)==2:                
#        d1 = pd.read_csv(d[0])
#        d2 = pd.read_csv(d[1])
#        st = 'D:\Codes\data\\unified_cases\\physio\\pp_'+x[0]+'.csv'
#        dfs = [d1, d2]
#        df = pd.concat(dfs,ignore_index=True)
#        df.to_csv(st,index=False)
#    else:
#        d1 = pd.read_csv(d[0])
#        d2 = pd.read_csv(d[1])
#        d3 = pd.read_csv(d[2])
#        st = 'D:\Codes\data\\unified_cases\\physio\\pp_'+x[0]+'.csv'
#        dfs = [d1, d2, d3]
#        df = pd.concat(dfs,ignore_index=True)
#        df.to_csv(st,index=False)

#mypath = 'D:\Codes\data\\unified_cases\\facereader'
#names = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#for name in names:
#    df = pd.read_csv('D:\Codes\data\\unified_cases\\facereader\\'+name)
#    df.loc[df['Condition'] == 'R', 'Condition'] = 'N'
#    df.loc[df['Condition'] == 'T', 'Condition'] = 'S'
#    df.loc[df['Condition'] == 'I', 'Condition'] = 'S'
#    df1 = df[df['Condition'] == 'N']
#    df2 = df[df['Condition'] == 'S']
#    dat = df[df.columns[[0,1,2,3,4]]]
#    df1_1 = df1.drop(df1.columns[[0,1,2,3,4]],axis=1)
#    df2_1 = df2.drop(df1.columns[[0,1,2,3,4]],axis=1)
#    means = df1_1.mean()
#    df3_1 = df1_1 - means 
#    df4_1 = df2_1 - means
#    df_concat = pd.concat([df3_1, df4_1],ignore_index=True)
#    df = pd.concat([dat, df_concat], axis=1)
#    df.to_csv('D:\Codes\data\\unified_cases\\baseline\\facereader\\'+name,index=False)
    

    
#mypath = 'D:\Codes\data\\unified_cases\\baseline\\physio'
#names = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#p4 = pd.DataFrame()
#dfs = []
#for i in range(0,len(names)):
#    p1 = pd.read_csv('D:\Codes\data\\unified_cases\\baseline\\physio\\'+names[i])
#    p4 = pd.concat([p4,p1], axis=0)
#    ## 2nd way, best
#    dfs.append(p1)
##p4 = p4.rename(columns={'Minute': 'Minutes'})
#p4 = p4.sort_values(['PP','C','Minutes'])
#p4 = p4.reset_index()
#p4 = p4.drop(['index'],axis=1)
#p4.to_csv(r"D:\Codes\data\\unified_cases\\physio_baseline_data.csv",index=False)