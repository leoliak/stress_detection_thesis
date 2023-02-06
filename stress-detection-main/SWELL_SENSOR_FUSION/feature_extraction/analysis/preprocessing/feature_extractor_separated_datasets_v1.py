"""
Diaxwrizei to dataset se 2 katigories, ta time feats kai ta frequency feats.
Kai sta 2 apothikeuei tis plirofories gia ton PP, to Condition kai to minute ths dokimis
"""

import pandas as pd
import numpy as np
import functions as fn
from tqdm import tqdm
import math
from collections import namedtuple
import re

##----------------------------------Initialization of FLAGS------------------------------------------##
## if True, when an outlier is found, the whole minute-window of data is not been taken into account
jump_for_outlier = False
filter_signal = False
resample = True

r = 'D:\Codes\dimpomatiki_v0_1\Separated_datasets\\time_feat_new_time\\'
path_data, onomata = fn.listerman(r, ".csv")
ran = len(onomata)
times = pd.read_csv('timestamps_altered.csv',dtype=str) #Read data from CSV datafile
for ii in range(0,ran):
    dataset1 = pd.read_csv(path_data[ii]) #Read data from CSV datafile
    path = onomata[ii]
#    if resample == True:
#        resample_rate = 4
#        df2 = dataset1.iloc[::resample_rate]
#        df2 = df2.reset_index()
#        df2 = df2.drop(['index'],1)
#        fs = int(2048/resample_rate)
#        hrw = 0.5*resample_rate
#        dataset1 = df2
### Input path of files
#    x = re.findall("\d+", path)
#    x1 = x[0]
#    x2 = x[4]
#    for g in range(0,times.shape[0]):
#        x3 = times.iloc[g][0]
#        x4 = times.iloc[g][1]
#        if x3==x1 and x4==x2:
#            hour = int(times.iloc[g][2])
#            minute = int(times.iloc[g][3])
#            second = int(times.iloc[g][4])
#            break
#    sec_remain = 60-second
#    new_data = dataset1[dataset1['time']>=sec_remain]
#    new_data.time = new_data.time - sec_remain
#    dataset1 = new_data.reset_index(drop=True)
#    print('Save new timeset...')
#    fn.write_to_csv2(dataset1,path,'time2')

    ##----------------------------------Parameters Setup------------------------------------------##
    hrw = 0.5 #One-sided window size, as proportion of the sampling frequency
    fs = 2048 #The example dataset was recorded at 2048Hz
    #Determine the upper limit for deciding a time-window is faulty
    # for rri=1200ms, it means bpm: 48
    high_rri = 1250 
    #Determine the lower limit for deciding a time-window is faulty 
    # for rri=400ms, it means bpm: 150
    low_rri = 400
    start_point = 0
    feats = {}
    time_f = {}
    freq_f = {}
    time = {'Minutes':[], 'PP':[], 'C':[], 'Condition':[]}
    
    ##--------------------------Initialization of Frequency Analysis Input Arguments--------------------------##
    VlfBand = namedtuple("Vlf_band", ["low", "high"])
    LfBand = namedtuple("Lf_band", ["low", "high"])
    HfBand = namedtuple("Hf_band", ["low", "high"])
    vlf_band = VlfBand(0.003, 0.04)
    lf_band = LfBand(0.04, 0.15)
    hf_band = HfBand(0.15, 0.40)
    method = "welch"
    samp_freq = 4
    int_method = "cubic"
    ## Counter to check between 2 methods of calculating bpm
    counter = 0
    if filter_signal == True:
        temp = fn.butter_lowpass_filter(dataset1.heart,2.5,2048.0,3)
        dataset1.heart = temp
        
    ## Dynamically set limiter for peak detection
#    if resample == True:
#        resample_rate = 4
#        df2 = dataset1.iloc[::resample_rate]
#        df2 = df2.reset_index()
#        df2 = df2.drop(['index'],1)
#        fs = int(2048/resample_rate)
#        hrw = 0.5*resample_rate
#        dataset1 = df2
#    dataset1['heart'] = dataset1['heart']*1000
#    dataset1['skin'] = dataset1['skin']*1000
    fs2 = fn.sampling_frequency(dataset1.time)
    participant, section, condition = fn.participant_finder(path)
    minutes = math.floor(dataset1.time[len(dataset1.time)-1]/60)
    for i in tqdm(range(1,int(minutes+1))):
        end_point = i*60*fs - 1
        if end_point>=len(dataset1): break
        dataset = dataset1[start_point:end_point]
        cardio = dataset.heart
        local_mean = np.mean(cardio)
        local_sd = np.std(cardio)
        if filter_signal == True:
            limiter = 1*local_mean + 1.25*local_sd
        else:
            limiter = 1*local_mean + 1*local_sd
                    
        ybeat, peaklist, timepeaklist = fn.peak_detector(dataset,start_point,hrw,fs,limiter)
        ########################################################################################################
        ### Calculate features [time_domain, frequency_domain]
        bpm = len([ii for ii in timepeaklist if ii <=i*60 and ii>60*(i - 1)])
        RR_list_all = fn.rr_dist_calc(timepeaklist)
        
        
        ### Add extra argument {do_print=1} into fn.correct_faulty_beats() in order to print outliers
        ### If jump_for_outlier is False, you have to determine what mode it should correct RR Intervals
        ### Mode 0: Simply reject them
        ### Mode 1: Interpolate the incorrect values
        ### Perc: Keeps only the 90% of HRV`s distribution for specific window
        ## By default mode 0 is chosen, otherwise type mode=1 into function`s arguments
        if jump_for_outlier == True:
            continue
        else:
            RR_list, num_of_outliers, outliers_list = fn.correct_faulty_beats(RR_list_all, high_rri, low_rri, mode=0, perc=False)
        
        
        ##----------------------------Feature Extraction----------------------------##
        # Calculate Time Features
        time_feats= fn.calc_time_features(RR_list, dataset["skin"], bpm)
        fn.plot_psd(RR_list.to_list)
        
        freq_feats = fn.get_frequency_domain_features(RR_list.tolist(), method, samp_freq, int_method, vlf_band, lf_band, hf_band)
        fr_time = {**time_feats}
        
        if i==1:
            time_f.update(fr_time)
            listes_t = list(time_f.keys())
        else:
            for title in listes_t:
                time_f[title] = np.append(time_f[title], fr_time[title])
        
        fr_freq = {**freq_feats}
        if i==1:
            freq_f.update(fr_freq)
            listes_f = list(freq_f.keys())
        else:
            for title in listes_f:
                freq_f[title] = np.append(freq_f[title], fr_freq[title])
        
        all_feats = fr_time = {**time_feats, **freq_feats}
        if i==1:
            feats.update(all_feats)
            listes = list(feats.keys())
        else:
            for title in listes:
                feats[title] = np.append(feats[title], all_feats[title])
        
        ########################################################################################################
        # Print results and graphs
        # Add extra argument {select='all'} into fn.printer() in order to print all time_feats
        
        fn.printer(i, time_feats)
        fn.plotter(i, dataset.time, dataset.heart, timepeaklist, ybeat, limiter,participant,condition)
        start_point = end_point + 1
        time['Minutes'].append(i)
        time['PP'].append(participant)
        time['C'].append(section)
        if i<=8:
            time['Condition'].append('R')
        else:
            time['Condition'].append(condition)
        if bpm==time_feats['bpm']:  counter += 1
    
    print('End of process')
    print('Total time of task:', minutes,'minutes')
    print('Average of HR is:', round(sum(time_f['bpm'])/minutes,2))
    # Function call for create dataset using time in minutes, HR and SCL calculated 
    total_time = {**time, **time_f}
    total_freq = {**time, **freq_f}
    print('Create Dataset File...')
    fn.write_to_csv2(total_time,path,'time2')
    fn.write_to_csv2(total_freq, path,'frequency')

    total = {**time, **feats}
    print('Create Dataset File...')
    fn.write_to_csv2(total, path, 'overall')


#### Write new dataset
write_new = False
if(write_new == True): 
    print('Writing new dataset...')
    r = 'D:\Codes\dimpomatiki_v0_1\dataset\\'
    path_data, onomata = fn.listerman(r, ".csv")
    dfs = []
    for i in range(0,len(onomata)):
        p1 = pd.read_csv(path_data[i])
        dfs.append(p1)
    p4 = pd.concat(dfs, ignore_index=True)
    p4 = p4.sort_values(['PP','C','Minutes'])
    p4.to_csv("D:\Codes\dimpomatiki_v0_1\data_2.csv")
print('-------------- END --------------')