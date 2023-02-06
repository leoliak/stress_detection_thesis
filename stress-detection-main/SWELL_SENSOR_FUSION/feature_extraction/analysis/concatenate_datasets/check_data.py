import pandas as pd
import numpy as np
import math
import re
from os import listdir
from os.path import isfile, join


#'''
#  Edw prosarmozw ta data pou thelw sta time_feat data mou kai ta apothikeuw san ksexwrista arxeia 
#  gia kathe participant
#'''
#mypath = 'D:\Codes\data\\physio\\data_freq'
#names2 = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#
#mypath = 'D:\Codes\data\\physio\\data_time'
#names = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#counter = 0
#counter2 = 0
#counter3 = 0
#d1 = []
#d2 = []
#d3 = []
#for i in range(0,len(names)):
#    x = re.findall("\d+", names[i])
#    pp = x[0]
#    c = x[1]
#    d10 = pd.read_csv('D:\Codes\data\\physio\\data_time\\'+names[i])
#    d10 = d10.drop(['Condition'],axis=1)
#    le_1 = d10.shape[0]
#    found = 0
#    for t in range(0,len(names2)):
#        x2 = re.findall("\d+", names2[t])
#        if x2[0]==pp and x2[1]==c:
#            d20 = pd.read_csv('D:\Codes\data\\physio\\data_freq\\'+names2[t])
##            d20 = d20.drop(['Minutes','PP'],axis=1)
#            print('Vrethike idio')
#            for_delete = []
#            l1 = d10.shape[0]
##            if l1<l2:
##                print('mpempa')
##                for g in range(0,d10.shape[0]):
##                    g1 = d10.iloc[g][4]
##                    diff = 0
##                    for h in range(0,d20.shape[0]):
##                        g2 = d20.iloc[h][4]
##                        if g1==g2:  
##                            for_delete.append(g)
##            new = d10[['timestamp']].copy()
##            for ii in range(0,new.shape[0]):
##                x = new.iloc[ii][0]
##                x1 = x[9:]
##                new.at[ii, 'timestamp'] = x1
##            sav = d20.timestamp
##            d20.insert(5, "timestamp2", sav, True) 
##            for ii in range(0,d20.shape[0]):
##                x = d20.iloc[ii][4]
##                x1 = x[9:]
##                d20.at[ii,'timestamp'] = x1
#            p4 = pd.merge(d10, d20, on=['timestamp','PP','Minutes','C'])
#            l2 = p4.shape[0]
#            if l1==l2:
#                counter+=1
#            elif l1<l2:
#                d1.append(x2)
#                print('megalitero to facereader kata:', l2-l1)
#            else:
#                d2.append(x2)
#                print('mikroterp to facereader kata:', l1-l2)
##            d20.drop(df.index[for_delete],inplace=True)
#            lab = p4.Condition
#            p4 = p4.drop(['Condition'],axis=1)
#            p4.insert(4,'Condition',lab,True)
#            p4.to_csv(r'D:\Codes\data\\physio\\whole\\data_'+names2[t], index=False)
#            break
#
#mypath = 'D:\Codes\data\\facereader\\'
#onomata = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#
#p4 = pd.DataFrame()
#dfs = []
#for i in range(0,len(onomata)):
#    p1 = pd.read_csv('D:\Codes\data\\facereader\\'+onomata[i])
#    p4 = pd.concat([p4,p1], axis=0)
#    ## 2nd way, best
#    dfs.append(p1)
##df_whole = pd.concat(dfs,ignore_index=True)
###df_whole = df_whole.rename(columns={'Minute': 'Minutes'})
##df_whole = df_whole.sort_values(['PP','C','Minutes'])
##p4 = p4.rename(columns={'Minute': 'Minutes'})
#p4 = p4.sort_values(['PP','C','Minute'])
#p4 = p4.reset_index()
#p4 = p4.drop(['index'],axis=1)
#p4.to_csv(r"D:\Codes\data\\facereader_data_new.csv",index=False)


#####################################################################################################
'''
  Edw prosarmozw ta data pou thelw sta time_feat data mou kai ta apothikeuw san ksexwrista arxeia 
  gia kathe participant
'''
mypath = 'D:\Codes\Dedomena_xristwn_me_timestamps\\kinect_data\\kin_new'
names2 = [f for f in listdir(mypath) if isfile(join(mypath, f))]

mypath = 'D:\Codes\Dedomena_xristwn_me_timestamps\\data_time'
names = [f for f in listdir(mypath) if isfile(join(mypath, f))]
counter = 0
counter2 = 0
counter3 = 0
d1 = []
d2 = []
d3 = []
for i in range(0,len(names)):
    x = re.findall("\d+", names[i])
    pp = x[0]
    c = x[1]
    d10 = pd.read_csv('data_time\\'+names[i])
    le_1 = d10.shape[0]
    found = 0
    for t in range(0,len(names2)):
        x2 = re.findall("\d+", names2[t])
        if x2[0]==pp and x2[1]==c:
            d20 = pd.read_csv('kinect_data\\kin_new\\'+names2[t])
            print('Vrethike idio')
            for_delete = []
            l1 = d10.shape[0]
#            if l1<l2:
#                print('mpempa')
#                for g in range(0,d10.shape[0]):
#                    g1 = d10.iloc[g][4]
#                    diff = 0
#                    for h in range(0,d20.shape[0]):
#                        g2 = d20.iloc[h][4]
#                        if g1==g2:  
#                            for_delete.append(g)
            new = d10[['timestamp']].copy()
            for ii in range(0,new.shape[0]):
                x = new.iloc[ii][0]
                x1 = x[9:]
                new.at[ii, 'timestamp'] = x1
            sav = d20.timestamp
            d20.insert(5, "timestamp2", sav, True) 
            for ii in range(0,d20.shape[0]):
                x = d20.iloc[ii][4]
                x1 = x[9:]
                d20.at[ii,'timestamp'] = x1
            p4 = pd.merge(new, d20, on='timestamp')
            l2 = p4.shape[0]
            if l1==l2:
                counter+=1
            elif l1<l2:
                d1.append(x2)
                print('megalitero to facereader kata:', l2-l1)
            else:
                d2.append(x2)
                print('mikroterp to facereader kata:', l1-l2)
#            d20.drop(df.index[for_delete],inplace=True)
            p4.to_csv(r'kin_data\\new_'+names2[t], index=False)
            break

#####################################################################################################
'''
   Gia kathe fakelo me katagrafes dimiourgw enniaia dataset gia kathe sensor
'''

mypath = 'D:\Codes\dedomena_xristwn_me_timestamps\\data_freq'
names = [f for f in listdir(mypath) if isfile(join(mypath, f))]

mypath = 'D:\Codes\dedomena_xristwn_me_timestamps\\data_time'
names2 = [f for f in listdir(mypath) if isfile(join(mypath, f))]

mypath = 'D:\Codes\dedomena_xristwn_me_timestamps\\facereader_data'
names3 = [f for f in listdir(mypath) if isfile(join(mypath, f))]

mypath = 'D:\Codes\dedomena_xristwn_me_timestamps\\kin_data'
names4 = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#df1 = pd.read_csv('data_freq\\'+names[1])
#df2 = pd.read_csv('data_time\\'+names2[1])
#df3 = pd.read_csv('facereader_data\\'+names3[1])

write_new = False

p4 = pd.DataFrame()
dfs = []
for i in range(0,len(names)):
    p1 = pd.read_csv('data_freq\\'+names[i])
    p4 = pd.concat([p4,p1], axis=0)
    ## 2nd way, best
    dfs.append(p1)
#p4 = p4.rename(columns={'Minute': 'Minutes'})
p4 = p4.sort_values(['PP','C','Minutes'])
p4 = p4.reset_index()
p4 = p4.drop(['index'],axis=1)
if write_new == True: p4.to_csv(r"D:\Codes\Dedomena_xristwn_me_timestamps\\dokimes\\frequency_data.csv",index=False)
df_freq = p4

p4 = pd.DataFrame()
dfs = []
for i in range(0,len(names2)):
    p1 = pd.read_csv('data_time\\'+names2[i])
    p4 = pd.concat([p4,p1], axis=0)
    ## 2nd way, best
    dfs.append(p1)
#p4 = p4.rename(columns={'Minute': 'Minutes'})
p4 = p4.sort_values(['PP','C','Minutes'])
p4 = p4.reset_index()
p4 = p4.drop(['index'],axis=1)
if write_new == True: p4.to_csv(r"D:\Codes\Dedomena_xristwn_me_timestamps\\dokimes\\time_data.csv",index=False)
df_time = p4

p4 = pd.DataFrame()
dfs = []
for i in range(0,len(names3)):
    p1 = pd.read_csv('facereader_data\\'+names3[i])
    p4 = pd.concat([p4,p1], axis=0)
    ## 2nd way, best
    dfs.append(p1)
p4 = p4.rename(columns={'Minute': 'Minutes'})
p4 = p4.sort_values(['PP','C','Minutes'])
p4 = p4.reset_index()
p4 = p4.drop(['index'],axis=1)
if write_new == True: p4.to_csv(r"D:\Codes\Dedomena_xristwn_me_timestamps\\dokimes\\face_data.csv",index=False)
df_face = p4

p4 = pd.DataFrame()
dfs = []
for i in range(0,len(names4)):
    p1 = pd.read_csv('kin_data\\'+names4[i])
    p4 = pd.concat([p4,p1], axis=0)
    ## 2nd way, best
    dfs.append(p1)
p4 = p4.rename(columns={'Minute': 'Minutes'})
p4 = p4.sort_values(['PP','C','Minutes'])
p4 = p4.reset_index()
p4 = p4.drop(['index'],axis=1)
if write_new == True: p4.to_csv(r"D:\Codes\Dedomena_xristwn_me_timestamps\\dokimes\\kinect_data.csv",index=False)
df_kin = p4
df_kin = df_kin.astype({"timestamp": str})

sav = df_time.timestamp
df_time.insert(5, "timestamp2", sav, True) 
for ii in range(0,df_time.shape[0]):
    x = df_time.iloc[ii][4]
    x1 = x[9:]
    df_time.at[ii,'timestamp'] = x1

sav = df_freq.timestamp
df_freq.insert(5, "timestamp2", sav, True) 
for ii in range(0,df_freq.shape[0]):
    x = df_freq.iloc[ii][4]
    x1 = x[9:]
    df_freq.at[ii,'timestamp'] = x1

sav = df_face.timestamp
df_face.insert(5, "timestamp2", sav, True) 
for ii in range(0,df_face.shape[0]):
    x = df_face.iloc[ii][0]
    x1 = x[9:]
    df_face.at[ii,'timestamp'] = x1

t = pd.merge(df_time,df_freq,on=['PP','C','timestamp'])
t = t.drop(['Minutes_x','timestamp2_x','Minutes_y','Condition_x'],axis=1)
t2 = pd.merge(t,df_face, on=['PP','C','timestamp'])
t2 = t2.drop(['timestamp2','Condition','Minutes'],axis=1)
t3 = pd.merge(t2,df_kin,on=['PP','C','timestamp'])
t3 = t3.drop(['Minutes','Condition','timestamp2','timestamp'],axis=1)

t3 = t3.rename(columns={'timestamp2_y': 'timestamp','Condition_y': 'Condition'})
con = t3.Condition
ti = t3.timestamp
t3 = t3.drop(['Condition','timestamp'],axis=1)
t3.insert(2,'timestamp',ti,True)
t3.insert(3,'Condition',con,True)

m = []
for col in t3.columns: 
    m.append(col)

t3.to_csv(r"D:\Codes\Dedomena_xristwn_me_timestamps\\all_in_dataset.csv",index=False)