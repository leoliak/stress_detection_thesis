#%% Initialization of parameters
"""
Created on Wed Sep 23 21:07:18 2020

@author: leonidas liakopoulos
"""

import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import spectrogram
import heartpy as hp
import pdb
import spec





#%% DATA PREPARATION

master = "ECG"
master2 = "data_3_extend_" + master.lower()

folder_stress_plot = master + "/" + master2 + "_full/plot_stress"
os.makedirs(folder_stress_plot, exist_ok=True)
folder_base_plot = master + "/" + master2 + "_full/plot_baseline"
os.makedirs(folder_base_plot, exist_ok=True)
folder_medi_plot = master + "/" + master2 + "_full/plot_meditation"
os.makedirs(folder_medi_plot, exist_ok=True)

folder_stress = master + "/" + master2 + "_full/spectrogram_stress"
os.makedirs(folder_stress, exist_ok=True)
folder_base = master + "/" + master2 + "_full/spectrogram_baseline"
os.makedirs(folder_base, exist_ok=True)
folder_medi = master + "/" + master2 + "_full/spectrogram_meditation"
os.makedirs(folder_medi, exist_ok=True)

folder_stress_data = master + "/" + master2 + "_matplot/spectrogram_stress"
os.makedirs(folder_stress_data, exist_ok=True)
folder_base_data = master + "/" + master2 + "_matplot/spectrogram_baseline"
os.makedirs(folder_base_data, exist_ok=True)
folder_medi_data = master + "/" + master2 + "_matplot/spectrogram_meditation"
os.makedirs(folder_medi_data, exist_ok=True)

folder_stress_data_mesh = master + "/" + master2 + "_plotmesh/spectrogram_stress"
os.makedirs(folder_stress_data_mesh, exist_ok=True)
folder_base_data_mesh = master + "/" + master2 + "_plotmesh/spectrogram_baseline"
os.makedirs(folder_base_data_mesh, exist_ok=True)
folder_medi_data_mesh = master + "/" + master2 + "_plotmesh/spectrogram_meditation"
os.makedirs(folder_medi_data_mesh, exist_ok=True)





# 
participants = ["S2", "S3", "S4", "S5", "S6", "S7",
                "S8", "S9", "S10", "S11", "S13", 
                "S14", "S15", "S16", "S17"]


fl = False
split = True

if fl:
    for participant in participants:
        print("Participant id: {}".format(participant))
        path = "WESAD/{}/{}.pkl".format(participant, participant)
        with open(path, 'rb') as f:
            data = pkl.load(f, encoding="latin1")
            
        ch_sig = data['signal']['chest']
        wr_sig = data['signal']['wrist']
        labels = data["label"]
        print(ch_sig.keys())
        signal = ch_sig["ECG"]
        signal_base = signal[labels == 1]
        signal_stress = signal[labels == 2]
        signal_medite = signal[labels == 4]
        
        list_run = [signal_base, signal_stress, signal_medite]
        for i, signal_ in enumerate(list_run):
            if i == 0:
                folder = folder_base
                folder_data_matplot = folder_base_data
                folder_data_mesh = folder_base_data_mesh
                folder_plot = folder_base_plot
            elif i == 1:
                folder = folder_stress
                folder_data_matplot = folder_stress_data
                folder_data_mesh = folder_stress_data_mesh
                folder_plot = folder_stress_plot
            else:
                folder = folder_medi
                folder_data_matplot = folder_medi_data
                folder_data_mesh = folder_medi_data_mesh
                folder_plot = folder_medi_plot
                
            counter = 0
            minutes_ = float(int(signal_.shape[0]/(700*60)))
            sam = 700*60
            
            for min_ in np.arange(0.0, minutes_, 0.1):
                min_ = float(round(min_, 1))
                start_sig = int(min_*700*60)
                end_sig = int((min_+1)*700*60)
                if end_sig > signal_.shape[0]: break
                sig = signal_[start_sig:end_sig]
                
                print("Minute examined: {} -> {}".format(min_, float(sig.shape[0]/700)))
                plt.plot(sig)
                plt.savefig(os.path.join(folder_plot, participant + "_" + str(counter) + ".jpg"), bbox_inches='tight')
                plt.close()
                

                wav_spectrogram = spec.spectro_pretty(sig.flatten(), 
                                                    nfft=2048)
                fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,4))
                cax = ax.matshow(np.transpose(wav_spectrogram), 
                                                interpolation='nearest', 
                                                aspect='auto', 
                                                cmap=plt.cm.afmhot, 
                                                origin='lower')
                plt.savefig(os.path.join(folder_data_mesh, participant + "_" + str(counter) + ".jpg"), bbox_inches='tight')

                fig.colorbar(cax)
                plt.title('Original Spectrogram')
                plt.savefig(os.path.join(folder, participant + "_" + str(counter) + ".jpg"))
                plt.axis('off')
                plt.close()

                plt.specgram(sig.flatten(), 
                            Fs=700, 
                            NFFT=2048, 
                            cmap=plt.cm.afmhot, 
                            noverlap=512)
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [sec]')
                plt.savefig(os.path.join(folder, participant + "_" + str(counter) + ".jpg"))
                plt.axis('off')
                plt.savefig(os.path.join(folder_data_matplot, participant + "_" + str(counter) + ".jpg"), bbox_inches='tight')
                plt.close()
                counter += 1



import shutil
import random


if split:
    data_split = 0.8
    dd = "ECG/datasets/dataset_3_extended"
    if os.path.exists(dd):
        shutil.rmtree(dd)

    data_211 = dd + "/train/no_stress"
    os.makedirs(data_211, exist_ok=True)
    data_212 = dd + "/train/stress"
    os.makedirs(data_212, exist_ok=True)

    data_221 = dd + "/eval/no_stress"
    os.makedirs(data_221, exist_ok=True)
    data_222 = dd + "/eval/stress"
    os.makedirs(data_222, exist_ok=True)

    x1 = "ECG/data_3_extend_ecg_matplot/spectrogram_stress"
    x2 = "ECG/data_3_extend_ecg_matplot/spectrogram_baseline"

    lx1 = os.listdir(x1)
    lx2_ = os.listdir(x2)
    lx2 = random.sample(lx2_, len(lx1))

    px1 = int(data_split*len(lx1))
    px2 = int(data_split*len(lx2))

    sx1 = random.sample(lx1, px1)
    sx2 = random.sample(lx2, px1)


    ## Organise stress samples
    for i, im in enumerate(sx1):
        src  = x1 + "/" + im
        dst = data_212 + "/" + im
        shutil.copyfile(src, dst)
    for i, im in enumerate(lx1):
        if im in sx1: continue
        src  = x1 + "/" + im
        dst = data_222 + "/" + im
        shutil.copyfile(src, dst)

    ## Organise neutral samples
    for i, im in enumerate(sx2):
        src  = x2 + "/" + im
        dst = data_211 + "/" + im
        shutil.copyfile(src, dst)
    for i, im in enumerate(lx2):
        if im in sx2: continue
        src  = x2 + "/" + im
        dst = data_221 + "/" + im
        shutil.copyfile(src, dst)

