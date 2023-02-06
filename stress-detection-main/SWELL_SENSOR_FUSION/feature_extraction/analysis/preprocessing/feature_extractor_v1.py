"""
Ypologizei epimerous dataset poy periexoun gia kathe katagrafi ola ta feats poy mporw na eksagw. Sto telos
yparxei kwdikas pou kanei merge ola ta dataset se ena enniaio.
"""

import pandas as pd
import numpy as np
import functions as fn
from tqdm import tqdm
import math
from collections import namedtuple
import os, sys
import custom_spectrogram as spec

##----------------------------------Initialization of FLAGS------------------------------------------##
## if True, when an outlier is found, the whole minute-window of data is not been taken into account
jump_for_outlier = False
filter_signal = True
downsample = False


## Input path of files   
##----------------------------------Parameters Setup------------------------------------------##
hrw = 0.5 #One-sided window size, as proportion of the sampling frequency
fs = 2048 #The example dataset was recorded at 2048Hz
high_rri = 1200 #Determine the upper limit for deciding a time-window is faulty
low_rri = 450 #Determine the lower limit for deciding a time-window is faulty 

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
spectro_extraction = False

r = '../Datasets_txt'
ran = os.listdir(r)
files_ecg = [os.path.join(r, x) for x in ran if "heart" in x]
files_eda = [os.path.join(r, x) for x in ran if "skin" in x]
files_time = [os.path.join(r, x) for x in ran if "time" in x]

files_ecg.sort()
files_eda.sort()
files_time.sort()

RES_heart = "results_ecg"
os.makedirs(RES_heart, exist_ok=True)

RES_DATASET = "feature_datasets"
os.makedirs(RES_DATASET, exist_ok=True)

if spectro_extraction:
    RES_spec = "scectrograms/results_ecg_spec"
    os.makedirs(RES_spec, exist_ok=True)

    RES_SP_HRN = "scectrograms/results_hrv"
    os.makedirs(RES_SP_HRN, exist_ok=True)

    RES_SP_HRN_C = "scectrograms/results_hrv_custom"
    os.makedirs(RES_SP_HRN_C, exist_ok=True)



for ii, (ecg_, eda_, time_) in enumerate(zip(files_ecg, files_eda, files_time)):
    print("Read ECG...")
    data_ecg = []
    f = open(ecg_, "r")
    for i, x in enumerate(f):
        data_instance = x[:-2]
        if "heart" in data_instance: continue
        data_ecg.append(float(data_instance))
    data_ecg = np.array(data_ecg)

    print("Read EDA...")
    data_eda = []
    f = open(eda_, "r")
    for i, x in enumerate(f):
        data_instance = x[:-2]
        if "skin" in data_instance: continue
        data_eda.append(float(data_instance))
    data_eda = np.array(data_eda)

    print("Read time...")
    data_time = []
    f = open(time_, "r")
    for i, x in enumerate(f):
        data_instance = x[:-2]
        if "time" in data_instance: continue
        data_time.append(float(data_instance))
    data_time = np.array(data_time)


    ## Counter to check between 2 methods of calculating bpm
    counter = 0
    
    ## Section for signal filtering
    if filter_signal == True:
        temp = fn.butter_lowpass_filter(data_ecg, 2.5,2048.0,3)
        data_ecg = temp
    
    ## Dynamically set limiter for peak detection
    fs2 = fn.sampling_frequency(data_time)
    
    ## For each recording it extracts participant id and condition
    participant, section, condition = fn.participant_finder_from_txt_files(ecg_)
    print(participant, section, condition)
    feature_extraction_dataset_ = RES_DATASET + "/{}_{}_{}.csv".format(participant, section, condition)
    minutes = math.floor(data_time[len(data_ecg)-1]/60)
    
    start_point = 0
    feats = {}
    time = {'Minutes':[], 'PP':[], 'C':[], 'Condition':[]}
    for i in tqdm(range(1,int(minutes+1))):
        end_point = i*60*2048 - 1
    
        if end_point>= data_ecg.shape[0]: break
        window_time = data_time[start_point:end_point]
        cardio = data_ecg[start_point:end_point]*10
        window_eda = data_eda[start_point:end_point]

        ## Set dynamically the cardio limit for peak detection
        local_mean = np.mean(cardio)
        local_sd = np.std(cardio)
        if filter_signal == True:
            limiter = 1*local_mean + 1.25*local_sd
        else:
            limiter = 1.5*local_mean + 1.5*local_sd
        ybeat, peaklist, timepeaklist = fn.peak_detector_3(cardio, window_time, start_point, hrw, fs, limiter)
        
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
        try:
            time_feats= fn.calc_time_features_2(RR_list, window_eda, bpm)
            freq_feats = fn.get_frequency_domain_features(RR_list.tolist(), method, samp_freq, int_method, vlf_band, lf_band, hf_band)
            fr = {**time_feats, **freq_feats}
        except:
            import pdb; pdb.set_trace()
            continue
        if i==1:
            feats.update(fr)
            listes = list(feats.keys())
        else:
            for title in listes:
                feats[title] = np.append(feats[title], fr[title])
        
        ########################################################################################################
        
        # Print results and graphs
        # Add extra argument {select='all'} into fn.printer() in order to print all time_feats
        
        fn.printer(i, time_feats)
        fn.plotter(i, window_time, cardio, timepeaklist, ybeat, limiter, participant, condition, RES_heart)
        
        rr_i = RR_list.tolist()

        if spectro_extraction:
            fn.spectro(cardio, RES_SP_HRN, participant, condition, i)
            spec.spectro_pretty(cardio, RES_SP_HRN_C, participant, condition, i)
            fn.spectrogram(cardio*2, 512, RES_spec, participant, condition, i)

        start_point = end_point + 1
        time['Minutes'].append(i)
        time['PP'].append(participant)
        time['C'].append(section)
        if i<=6:
            time['Condition'].append('R')
        else:
            time['Condition'].append(condition)
        if bpm==time_feats['bpm']:  counter += 1

    print('End of process')
    print('Total time of task:', minutes,'minutes')
    print('Average of HR is:', round(sum(feats['bpm'])/minutes,2))
    
    # Function call for create dataset using time in minutes, HR and SCL calculated 
    total = {**time, **feats}
    print('Create Dataset File...')
    fn.write_to_csv(total, feature_extraction_dataset_)
    
    
