import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import interpolate #Import the interpolate function from SciPy
from scipy import signal
from scipy.signal import butter, lfilter #Import the extra module required
import os
from os import listdir
from os.path import isfile, join
import re
import imageio
import scipy

import warnings
warnings.filterwarnings('ignore')



def plotter(i, time, heart, timepeaklist, ybeat, limiter, participant, con, path, pause = 0.2):
    plt.title("Detected peaks in minute:"+str(i)+" for PP:"+str(participant))
    plt.plot(time, heart, alpha=0.5, color='blue') #Plot semi-transparent HR
    plt.axhline(limiter, color='r', linestyle='dotted')
    #plt.plot(mov_avg, color ='green') #Plot moving average
    plt.scatter(timepeaklist, ybeat, color='red') #Plot detected peaks
    plt.savefig(path + '/' + str(participant) + '_c' + str(con) + 'minute_' + str(i) + '.png' )
    plt.close()


def dictionary_rounder(dictionary):
    for dict_value in dictionary:
        for k, v in dict_value.items():
            dict_value[k] = round(v, 4)


def spectrogram(sig, fs_, path, participant, con, i):
    # f, t, Sxx = scipy.signal.spectrogram(sig, fs=fs_, window=('tukey', 0.25), 
    #                                 nperseg=230, noverlap=0.5, return_onesided=True)
    # log_spectrogram = np.log(Sxx + 1)
    # cc1 = log_spectrogram[:, :]
    
    # plt.pcolormesh(t, f, Sxx, cmap="viridis", shading='gouraud')
    plt.specgram(sig.flatten(), 
                        Fs=700, 
                        NFFT=fs_, 
                        cmap=plt.cm.afmhot, 
                        noverlap=128)
    plt.axis('off')
    plt.savefig(path + '/' + str(participant) + '_c' + str(con) + '_minute_' + str(i) + '.png', bbox_inches='tight' )
    plt.close()


def participant_finder(st):
    x = re.findall("\d+", st)
    participant = x[0]
    part = x[4]
    list_b = ['1','3','5','7','9','11','13','15','17','20','22','24']
    if part == '1':
        condition = 'N'
    elif (part == '2') and (participant in list_b):
        condition = 'T'
    elif (part == '2') and (participant not in list_b):
        condition = 'I'
    elif (part == '3') and (participant in list_b):
        condition = 'I'
    elif (part == '3') and (participant not in list_b):
        condition = 'T'
    return participant, part, condition



def participant_finder_from_txt_files(st):
    s = os.path.basename(st)
    s1 = os.path.splitext(s)[0]
    x = s1.split("_")
    if len(x) == 4:
        participant = x[1]
        section = x[2].split("-")[3]
    elif len(x) == 5:
        participant = x[1]
        section = x[3]
    part = section.replace("c", "")
    list_b = ['1','3','5','7','9','11','13','15','17','20','22','24']
    if part == '1':
        condition = 'N'
    elif (part == '2') and (participant in list_b):
        condition = 'T'
    elif (part == '2') and (participant not in list_b):
        condition = 'I'
    elif (part == '3') and (participant in list_b):
        condition = 'I'
    elif (part == '3') and (participant not in list_b):
        condition = 'T'
    return participant, part, condition


def gif_cardio(i,participant,con):
    images = []
    for e in range(1,i+1):
        img_name = 'D:\Codes\dimpomatiki_v0_1\images\minute_' + str(e) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave('D:\Codes\dimpomatiki_v0_1\images\ecg_' +str(participant)+'_'+str(con)+'.gif', images, fps=4)


def printer(i,time_feats,select = "basic"):
    print('\nMinute:',i,', bpm:',time_feats['bpm'],', scl:',time_feats['Skin_mean'])
    if select=='all':
        print("RMSSD is:",time_feats['RMSSD'])
        print("Mean of RR-intervals is:",time_feats['Mean'])
        print("SD of RR-intervals is:",time_feats['Sd'])
        print("NN50 of RR-intervals is:",time_feats['NN50'])
        print("pNN50 of RR-intervals is:",time_feats['pNN50'],'%')
        print("NN25 of RR-intervals is:",time_feats['NN25'])
        print("pNN25 of RR-intervals is:",time_feats['pNN25'],'%')
        print("SDSD of RR-intervals is:",time_feats['SDSD'])
        print("Median of RR-intervals is:",time_feats['MedianNN'])
        print("Sd1 of RR-intervals is:",time_feats['SD1'])
        print("Sd2 of RR-intervals is:",time_feats['SD2'])
    return



def rr_dist_calc(timepeaklist):
    RR_list = []
    RR_list= np.diff(timepeaklist)*1000 #for milliseconds
    return RR_list



def sampling_frequency(dataset_time):
    sampletimer = [x for x in dataset_time] #dataset.timer is a ms counter with start of recording at '0'
    measured_fs = round(((len(sampletimer) / sampletimer[-1]))) #Divide total length of dataset by last timer entry.
    return measured_fs


def listerman(folder_path, type_of_file):
    lis = []
    c = []
    fv= []
    fv = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    x = os.path.dirname(os.path.abspath(__file__))
    for file in os.listdir(folder_path):
        if file.endswith(type_of_file):
            lis.append(os.path.join(x, file))
    for fi in lis:
        filename_w_ext = os.path.basename(fi)
        filename, file_extension = os.path.splitext(filename_w_ext)
        c.append(folder_path + filename_w_ext)
       # d.append(os.path.abspath(filename_w_ext))
    return c, fv


def correct_faulty_beats(RR_list, high_rri, low_rri, do_print=1, mode=0, perc=False):
    '''
    #### INSTRUCTIONS ####
    # mode-0: Reject all faulty RR Intervals
    # mode-1: Interpolate all rejected RR Intervals
    '''
    if perc == True:
        or1 = np.percentile(RR_list,5)
        or2 = np.percentile(RR_list,95)
        rr_intervals_cleaned = [ii for ii in RR_list if ii>or1 and ii<or2]
    else:
        rr_intervals_cleaned = [rri if high_rri >= rri >= low_rri else np.nan for rri in RR_list]
    series_cleaned = pd.Series(rr_intervals_cleaned)
    if mode == 0:
        cleanedList2 = [x for x in series_cleaned if str(x) != 'nan']
        cleanedList = pd.Series(cleanedList2)
    else:
        ending = len(rr_intervals_cleaned)-1
        ### Protection if nan is in beginning or ending of file
        if np.isnan(rr_intervals_cleaned[0])==True:
            if RR_list[0]>=high_rri: rr_intervals_cleaned[0] = high_rri
            if RR_list[0]<=low_rri: rr_intervals_cleaned[0] = low_rri
        if np.isnan(rr_intervals_cleaned[ending])==True:
            if RR_list[ending]>=high_rri: rr_intervals_cleaned[ending] = high_rri
            if RR_list[ending]<=low_rri: rr_intervals_cleaned[ending] = low_rri
        series_cleaned = pd.Series(rr_intervals_cleaned)
        cleanedList = series_cleaned.interpolate(method='linear')
    outliers_list = []
    for rri in RR_list:
        if high_rri >= rri >= low_rri:
            pass
        else:
            outliers_list.append(rri)
    nan_count = sum(np.isnan(rr_intervals_cleaned))
    if (do_print==1):
        if nan_count == 0:
            print(nan_count, "outlier(s) have been deleted.")
        else:
            print(nan_count, " outlier(s) have been deleted.")
            print("The outlier(s) value(s) are :", outliers_list)
        plt.title("HRV singal extracted")
        plt.ylabel("Time period (ms)")
        plt.xlabel("Number of accepted RR-Intervals")
        plt.plot(cleanedList, alpha=0.8, color='blue')
        plt.close()

    return cleanedList, nan_count, outliers_list


def calc_time_features(RR_list, skin, bpm1):
    RR_diff = np.diff(RR_list)
    
    # Basic statistics features
    vari = np.var(RR_list)
    mean_val = np.mean(RR_list)
    std_val = np.std(RR_list)
    medianNN = np.median(RR_list)
    
    # RR-diff list
    nn50 = np.add(len(np.where(abs(RR_diff) > 50)[0])/1.0,0.0)
    pnn50 = 100*(nn50/len(RR_list))
    
    nn25 = np.add(len(np.where(abs(RR_diff) > 25)[0])/1.0,0.0)
    pnn25 = 100*(nn25/len(RR_list))
    
    sdsd = np.std(RR_diff, ddof=1)
    sd1 = ((1 / np.sqrt(2)) * sdsd) #measures the width of poincare cloud
    sd2 = np.sqrt((2 * sdsd ** 2) - (0.5 * sdsd ** 2)) #measures the length of the poincare cloud
    
    rmssd = np.sqrt(sum(np.square(RR_diff))/(len(RR_list)-1)) #Take root of the mean of the list of squared differences
    
    ##skin feature extraction
    skin_mean = np.mean(skin)
    skin_deviation = np.std(skin)
    
    #skin_first_dif = round(sum(abs(np.diff(skin)))/(len(skin)-1), 4)
    # Heart Rate equivalent features
    timeDomainFeats = {'bpm':bpm1,'Mean': mean_val, 'Sd': std_val,
                       'var': vari,'SDSD': sdsd, 'NN50': nn50,
                       'pNN50': pnn50, 'NN25': nn25,
                       'pNN25': pnn25, 'SD1':sd1, 'SD2':sd2,
                       'MedianNN':medianNN, 'RMSSD':rmssd,
                       'skin_mean':skin_mean, 'skin_dev':skin_deviation}
    for dict_value in timeDomainFeats:
        c = timeDomainFeats.get(dict_value)
        timeDomainFeats[dict_value] = round(c, 4)
        if dict_value=='bpm': timeDomainFeats[dict_value] = round(c)
    return timeDomainFeats


def calc_time_features_2(RR_list, skin, bpm1):
    RR_diff = np.diff(RR_list)
    
    # Basic statistics features on HRV
    vari = np.var(RR_list)
    mean_val = np.mean(RR_list)
    std_val = np.std(RR_list)
    medianNN = np.median(RR_list)
    RR_range = max(RR_list) - min(RR_list)
    
    ## HRV Complex features
    nn50 = sum(np.abs(RR_diff) > 50)
    pnn50 = 100*(nn50/len(RR_list))
    nn25 = sum(np.abs(RR_diff) > 25)
    pnn25 = 100*(nn25/len(RR_list))
    sdsd = np.std(RR_diff, ddof=1)
    
    sd1 = ((1 / np.sqrt(2)) * sdsd) #measures the width of poincare cloud
    sd2 = np.sqrt((2 * sdsd ** 2) - (0.5 * sdsd ** 2)) #measures the length of the poincare cloud
    
    rmssd = np.sqrt(np.mean(RR_diff ** 2)) #Take root of the mean of the list of squared differences
    
    ## Skin features
    skin_mean = np.mean(skin)
    skin_std = np.std(skin)
    skin_var = np.var(skin)
    skin_max = np.max(skin)
    skin_min = np.min(skin)
    
    ## Heart Rate features
    hr_list = np.divide(60000, RR_list)
    mean_hr = np.mean(hr_list)
    min_hr = min(hr_list)
    max_hr = max(hr_list)
    min_max_diff = max_hr - min_hr
    std_hr = np.std(hr_list)
    timeDomainFeats = {'bpm':bpm1, 'HR_mean': mean_hr, 'HR_min':min_hr, 'HR_max':max_hr,
                       'HR_diff':min_max_diff, 'HR_std': std_hr,
                       'RR_Mean': mean_val, 'RR_Std': std_val,
                       'RR_Var': vari,'RR_range':RR_range, 'SDSD': sdsd, 'NN50': nn50,
                       'pNN50': pnn50, 'NN25': nn25,
                       'pNN25': pnn25, 'SD1':sd1, 'SD2':sd2,
                       'RR_Median':medianNN, 'RMSSD':rmssd,
                       'Skin_mean':skin_mean, 'Skin_dev':skin_std, 'Skin_var': skin_var,
                       'Skin_max':skin_max, 'Skin_min':skin_min}
    for dict_value in timeDomainFeats:
        c = timeDomainFeats.get(dict_value)
        timeDomainFeats[dict_value] = round(c, 3)
        if dict_value=='bpm': timeDomainFeats[dict_value] = round(c)
    return timeDomainFeats


def calc_hr_time_features(RR_list):
    heart_rate_list = np.divide(60000, RR_list)
    mean_hr = np.mean(heart_rate_list)
    min_hr = min(heart_rate_list)
    max_hr = max(heart_rate_list)
    std_hr = np.std(heart_rate_list)
    hr_feats = {'mean_hr': mean_hr, 'min_hr':min_hr,'max_hr':max_hr,'std_hr':std_hr}
    return hr_feats


def peak_detector(signal1,start_point, hrw, fs, upper_limit):
    signal = pd.Series(signal1.flatten())
    mov_avg = signal.rolling(int(hrw*fs)).mean() #Calculate moving average
    
    #Impute where moving average function returns NaN, which is the beginning of the signal where x hrw
    avg_hr = (np.mean(signal))
    mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]
    mov_avg = [x for x in mov_avg]
    heart_rollingmean = mov_avg #Append the moving average to the dataframe
    window = []
    peaklist = []
    listpos = start_point #We use a counter to move over the different data columns
    for datapoint in signal:
        rollingmean = heart_rollingmean[listpos] #Get local mean
        if (datapoint < rollingmean) and (len(window) < 1): #If no detectable R-complex activity -> do nothing
            listpos += 1
        elif (datapoint > rollingmean): #If signal comes above local mean, mark ROI
            window.append(datapoint)
            listpos += 1
        else: #If signal drops below local mean -> determine highest point
            beatposition = listpos - len(window) + (window.index(max(window))) #Notate the position of the point on the X-axis
            if(signal[beatposition]>upper_limit):
                peaklist.append(beatposition) #Add detected peak to list
            window = [] #Clear marked ROI
            listpos += 1
    ybeat = [signal[x] for x in peaklist] #Get the y-value of all peaks for plotting purposes

    return ybeat, peaklist


def create_timestamp_list(rr_ints, samp_freq, interpolation = False):
    """
    Creates corresponding time interval for all rr_ints
    """
    # Convert in seconds
    nni_tmstp = np.cumsum(rr_ints)/1000
    # Force to start at 0
    nni = nni_tmstp - nni_tmstp[0]
    if interpolation == True:
        temp = np.arange(0, nni[-1], 1/samp_freq)
        nni = temp
    return nni


def spectro(rr_i, path, participant, con, i):
    fs1 = 8
    timestamp_list = create_timestamp_list(rr_i, fs1)
    timestamps_interpolation = create_timestamp_list(rr_i, fs1, interpolation = True)

    # ---------- Interpolation of signal ---------- #
    # tck = scipy.interpolate.splrep(timestamp_list, rr_i, s=0)
    # rri_interp = scipy.interpolate.splev(timestamps_interpolation, tck, der=0)
    # funct = scipy.interpolate.interp1d(timestamp_list, rr_i, 'linear')
    # nn_i = funct(timestamps_interpolation)

    # ---------- Remove DC Component ---------- #
    # nn_i2 = nn_i - np.mean(nn_i)
    # f, t, Sxx = scipy.signal.spectrogram(rri_interp, fs=fs1)
    # plt.pcolormesh(t, f, Sxx)

    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.savefig(os.path.join(folder, participant + "_" + str(counter) + ".jpg"))
    # plt.axis('off')
    # plt.savefig(os.path.join(folder_data_matplot, participant + "_" + str(counter) + ".jpg"), bbox_inches='tight')
    # plt.close()
    Sxx, f, t, im = plt.specgram(rr_i, Fs=fs1, cmap=plt.cm.afmhot)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.savefig(path + '/' + str(participant) + '_c' + str(con) + '_minute_' + str(i) + '.png' )
    plt.close()




def peak_detector2(dataset,start_point, hrw, fs, upper_limit = 10):
    mov_avg = dataset['skin'].rolling(int(hrw*fs)).mean() #Calculate moving average
    #Impute where moving average function returns NaN, which is the beginning of the signal where x hrw
    avg_hr = (np.mean(dataset.skin))
    mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]
    mov_avg = [x for x in mov_avg]
    m_a = pd.Series(mov_avg)
    dataset['skin_rollingmean'] = m_a.values #Append the moving average to the dataframe
    window = []
    peaklist = []
    timepeaklist = []
    listpos = start_point #We use a counter to move over the different data columns
    for datapoint in dataset.skin:
        rollingmean = dataset.skin_rollingmean[listpos] #Get local mean
        if (datapoint < rollingmean) and (len(window) < 1): #If no detectable R-complex activity -> do nothing
            listpos += 1
        elif (datapoint > rollingmean): #If signal comes above local mean, mark ROI
            window.append(datapoint)
            listpos += 1
        else: #If signal drops below local mean -> determine highest point
            beatposition = listpos - len(window) + (window.index(max(window))) #Notate the position of the point on the X-axis
            if(dataset.skin[beatposition]>upper_limit):
                peaklist.append(beatposition) #Add detected peak to list
            window = [] #Clear marked ROI
            listpos += 1
    ybeat = [dataset.skin[x] for x in peaklist] #Get the y-value of all peaks for plotting purposes
    for ii in peaklist:
        timepeaklist.append(dataset.time[ii])
    return ybeat, peaklist, timepeaklist


def peak_detector_3(cardio_signal, window_time, start_point, hrw, fs, upper_limit):
    sig = pd.Series(cardio_signal)
    mov_avg = sig.rolling(int(hrw*fs)).mean() #Calculate moving average
    #Impute where moving average function returns NaN, which is the beginning of the signal where x hrw
    avg_hr = (np.mean(sig))
    mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]
    mov_avg = [x for x in mov_avg]
    m_a = pd.Series(mov_avg)
    heart_rollingmean = m_a.values #Append the moving average to the dataframe
    window = []
    peaklist = []
    timepeaklist = []
    # listpos = start_point #We use a counter to move over the different data columns
    listpos = 0
    for datapoint in sig:
        rollingmean = heart_rollingmean[listpos] #Get local mean
        if (datapoint < rollingmean) and (len(window) < 1): #If no detectable R-complex activity -> do nothing
            listpos += 1
        elif (datapoint > rollingmean): #If signal comes above local mean, mark ROI
            window.append(datapoint)
            listpos += 1
        else: #If signal drops below local mean -> determine highest point
            beatposition = listpos - len(window) + (window.index(max(window))) #Notate the position of the point on the X-axis
            if(sig[beatposition]>upper_limit):
                peaklist.append(beatposition) #Add detected peak to list
            window = [] #Clear marked ROI
            listpos += 1
    ybeat = [sig[x] for x in peaklist] #Get the y-value of all peaks for plotting purposes
    for ii in peaklist:
        timepeaklist.append(float(window_time[ii]))
    return ybeat, peaklist, timepeaklist


def detect_peaks(ecg_signal, threshold=0.95, qrs_filter=None):
    '''
    Peak detection algorithm using cross corrrelation and threshold 
    '''
    if qrs_filter is None:
        # create default qrs filter, which is just a part of the sine function
        t = np.linspace(1.5 * np.pi, 3.5 * np.pi, 15)
        qrs_filter = np.sin(t)
    
    # normalize data
    ecg_signal = (ecg_signal - ecg_signal.mean()) / ecg_signal.std()

    # calculate cross correlation
    similarity = np.correlate(ecg_signal, qrs_filter, mode="same")
    similarity = similarity / np.max(similarity)

    # return peaks (values in ms) using threshold
    return ecg_signal[similarity > threshold].index, similarity





def write_to_csv(d,path):
    features = pd.DataFrame(data=d)
    features.to_csv(path, index = False)
    return

def write_to_csv2(d,path,kind):
    features = pd.DataFrame(data=d)
    if kind=='frequency':
        p = os.path.join(r'D:\Codes\dimpomatiki_v0_1\Separated_datasets\freq_feat',path)
        features.to_csv(p, index = False)
    elif kind=='time2':
        p = os.path.join(r'D:\Codes\dimpomatiki_v0_1\Separated_datasets\time_feat_new_time',path)
        features.to_csv(p, index = False)
    else:
        p = os.path.join(r'D:\Codes\dimpomatiki_v0_1\Separated_datasets\time_feat_3',path)
        features.to_csv(p, index = False)
    return

def write_only_freq(d,path):
    features = pd.DataFrame(data=d)
    p = os.path.join(r'D:\Codes\dimpomatiki_v0_1\Separated_datasets\frequency_feat_windowed',path)
    features.to_csv(p, index = False)
    return

#def butter_lowpass(cutoff, fs, order=5):
#    nyq = 0.5 * fs #Nyquist frequeny is half the sampling frequency
#    normal_cutoff = cutoff / nyq
#    b, a = butter(order, normal_cutoff, btype='low', analog=False)
#    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=3):
    nyq = 0.5 * fs #Nyquist frequeny is half the sampling frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y


###################################################################################################################
##############################################################################################
# Named Tuple for different frequency bands
#VlfBand = namedtuple("Vlf_band", ["low", "high"])
#LfBand = namedtuple("Lf_band", ["low", "high"])
#HfBand = namedtuple("Hf_band", ["low", "high"])
#vlf_band = VlfBand(0.003, 0.04)
#lf_band = LfBand(0.04, 0.15)
#hf_band = HfBand(0.15, 0.40)
#method = "welch"
#samp_freq = 4
#int_method = "linear"

def get_frequency_domain_features(rr_interval_list, method, samp_freq, int_method, vlf_band, lf_band, hf_band):
    """
    Returns a dictionary containing frequency domain features for HRV analyses.
    To our knowledge, you might use this function on short term recordings, from 2 to 5 minutes  \
    window.
    Parameters
    ---------
    rr_interval_list : list
        list of Normal to Normal Interval
    method : str
        Method used to calculate the psd. Choice is Welch's method.
    samp_freq : int
        Frequency at which the signal is sampled. Common value range from 1 Hz to 10 Hz,
        by default set to 4 Hz.
    int_method : str
        kind of interpolation as a string, by default "linear".
    vlf_band : tuple
        Very low frequency bands for features extraction from power spectral density.
    lf_band : tuple
        Low frequency bands for features extraction from power spectral density.
    hf_band : tuple
        High frequency bands for features extraction from power spectral density.
    Returns
    ---------
    Notes
    ---------
    Details:
    - **total_power** : Total power density spectral
    - **vlf** : variance ( = power ) in HRV in the Very low Frequency (.003 to .04 Hz by default). \
    Reflect an intrinsic rhythm produced by the heart which is modulated primarily by sympathetic \
    activity.
    - **lf** : variance ( = power ) in HRV in the low Frequency (.04 to .15 Hz). Reflects a \
    mixture of sympathetic and parasympathetic activity, but in long-term recordings, it reflects \
    sympathetic activity and can be reduced by the beta-adrenergic antagonist propanolol.
    - **hf**: variance ( = power ) in HRV in the High Frequency (.15 to .40 Hz by default). \
    Reflects fast changes in beat-to-beat variability due to parasympathetic (vagal) activity. \
    Sometimes called the respiratory band because it corresponds to HRV changes related to the \
    respiratory cycle and can be increased by slow, deep breathing (about 6 or 7 breaths per \
    minute) and decreased by anticholinergic drugs or vagal blockade.
    - **lf_hf_ratio** : lf/hf ratio is sometimes used by some investigators as a quantitative \
    mirror of the sympatho/vagal balance.
    - **lfnu** : normalized lf power.
    - **hfnu** : normalized hf power.
    """

    # ----------  Compute frequency & Power spectral density of signal  ---------- #
    freq, psd = get_freq_psd_from_rr_interval_list(rr_interval_list, samp_freq, int_method)
    # ---------- Features calculation ---------- #
    freqency_domain_features = get_features_from_psd(freq, psd, vlf_band, lf_band, hf_band)
    return freqency_domain_features



def get_freq_psd_from_rr_interval_list(rr_interval_list, samp_freq, int_method):
    """
    Returns the frequency and power of the signal.
    ---------
    freq : list
        Frequency of the corresponding psd points.
    psd : list
        Power Spectral Density of the signal.
    """
    timestamp_list = create_newtime_list(rr_interval_list)
    
    # ---------- Interpolation of RR interal signal ---------- #
    funct = interpolate.interp1d(timestamp_list, rr_interval_list, int_method)
    timestamps_interpolation = create_newtime_list(rr_interval_list, interpolation = True)
    nn_interpolation = funct(timestamps_interpolation)

    # ---------- Remove DC Component ---------- #
    nn_normalized = nn_interpolation - np.mean(nn_interpolation)
    # print(nn_interpolation)
    # print(nn_normalized)

    #  --------- Compute Power Spectral Density  --------- #
    samp_freq = 4
    freq, psd = signal.welch(nn_normalized, samp_freq, window='hanning')
    
    ## Plot PSD diagram for each available frequency
    # plo = True
    # if plo==True:
    #     plt.figure(2)
    #     plt.clf()
    #     plt.semilogy(freq, psd)
    #     plt.xlabel('frequency [Hz]')
    #     plt.ylabel('PSD [V**2/Hz]')
    #     plt.show()
    #     plt.pause(0.2)
    return freq, psd


def create_newtime_list(rr_interval_list, samp_freq = 4, interpolation = False):
    """
    Creates corresponding time interval for all rr_interval_list
    """
    # Convert in seconds
    new_time = np.cumsum(rr_interval_list)/1000
    # Force to start at 0
    nni = new_time - new_time[0]
    if interpolation == True:
        temp = np.arange(0, nni[-1], 1/samp_freq)
        nni = temp
    return nni


def get_features_from_psd(freq, psd, vlf_band, lf_band, hf_band):
    """
    Computes frequency domain features from the power spectral decomposition.
    Parameters
    ---------
    freq : array
        Array of sample frequencies.
    psd : list
        Power spectral density or power spectrum.
    vlf_band : tuple
        Very low frequency bands for features extraction from power spectral density.
    lf_band : tuple
        Low frequency bands for features extraction from power spectral density.
    hf_band : tuple
        High frequency bands for features extraction from power spectral density.
    """

    # Calcul of indices between desired frequency bands
    vlf_indexes = np.logical_and(freq >= vlf_band[0], freq < vlf_band[1])
    lf_indexes = np.logical_and(freq >= lf_band[0], freq < lf_band[1])
    hf_indexes = np.logical_and(freq >= hf_band[0], freq < hf_band[1])

    # Integrate using the composite trapezoidal rule
    vlf = np.trapz(y=psd[vlf_indexes], x=freq[vlf_indexes])
    lf = np.trapz(y=psd[lf_indexes], x=freq[lf_indexes])
    hf = np.trapz(y=psd[hf_indexes], x=freq[hf_indexes])

    # total power & vlf : Feature often used for  "long term recordings" analysis
    total_power = vlf + lf + hf

    lf_hf_ratio = lf / hf
    lfnu = (lf / (total_power-vlf)) * 100
    hfnu = (hf / (total_power-vlf)) * 100
    
    frequency_domain_features = {
        'lf': lf,
        'ln_lf':  np.log(lf),
        'hf': hf,
        'ln_hf': np.log(hf),
        'lf_hf_ratio': lf_hf_ratio,
        'ln_lf_hf_ratio': np.log(lf_hf_ratio),
        'norm_lf': lfnu,
        'norm_hf': hfnu,
        'total_power': total_power,
        'vlf': vlf,
        'ln_vlf': np.log(vlf)
    }
    
    for dict_value in frequency_domain_features:
        c = frequency_domain_features.get(dict_value)
        frequency_domain_features[dict_value] = round(c, 2)
    return frequency_domain_features


from collections import namedtuple
from matplotlib import style

VlfBand = namedtuple("Vlf_band", ["low", "high"])
LfBand = namedtuple("Lf_band", ["low", "high"])
HfBand = namedtuple("Hf_band", ["low", "high"])


def plot_psd(nn_intervals, path, participant, con, i,
             method: str = "welch", sampling_frequency: int = 8,
             interpolation_method: str = "linear", vlf_band: namedtuple = VlfBand(0.003, 0.04),
             lf_band: namedtuple = LfBand(0.04, 0.15), hf_band: namedtuple = HfBand(0.15, 0.40)):
    """
    Function plotting the power spectral density of the NN Intervals.
    Arguments
    ---------
    nn_intervals : list
        list of Normal to Normal Interval.
    method : str
        Method used to calculate the psd. Choice are Welch's FFT (welch) or Lomb method (lomb).
    sampling_frequency : int
        frequence at which the signal is sampled. Common value range from 1 Hz to 10 Hz, by default
        set to 7 Hz. No need to specify if Lomb method is used.
    interpolation_method : str
        kind of interpolation as a string, by default "linear". No need to specify if lomb method is
        used.
    vlf_band : tuple
        Very low frequency bands for features extraction from power spectral density.
    lf_band : tuple
        Low frequency bands for features extraction from power spectral density.
    hf_band : tuple
        High frequency bands for features extraction from power spectral density.
    """

    freq, psd = get_freq_psd_from_rr_interval_list(nn_intervals, sampling_frequency, interpolation_method)

    # Calcul of indices between desired frequency bands
    vlf_indexes = np.logical_and(freq >= vlf_band[0], freq < vlf_band[1])
    lf_indexes = np.logical_and(freq >= lf_band[0], freq < lf_band[1])
    hf_indexes = np.logical_and(freq >= hf_band[0], freq < hf_band[1])
    frequency_band_index = [vlf_indexes, lf_indexes, hf_indexes]
    label_list = ["VLF component", "LF component", "HF component"]

    # Plot parameters
    style.use("seaborn-darkgrid")
    plt.figure(figsize=(12, 8))
    plt.xlabel("Frequency (Hz)", fontsize=15)
    plt.ylabel("PSD (s2/ Hz)", fontsize=15)

    if method == "lomb":
        plt.title("Lomb's periodogram", fontsize=20)
        for band_index, label in zip(frequency_band_index, label_list):
            plt.fill_between(freq[band_index], 0, psd[band_index] / (1000 * len(psd[band_index])), label=label)
        plt.legend(prop={"size": 15}, loc="best")

    elif method == "welch":
        plt.figure(4)
        plt.clf()
        plt.title("FFT Spectrum : Welch's periodogram", fontsize=20)
        for band_index, label in zip(frequency_band_index, label_list):
            plt.fill_between(freq[band_index], 0, psd[band_index] / (1000 * len(psd[band_index])), label=label)
        plt.legend(prop={"size": 15}, loc="best")
        plt.xlim(0, hf_band[1])
    else:
        raise ValueError("Not a valid method. Choose between 'lomb' and 'welch'")

    plt.savefig(path + '/' + str(participant) + '_c' + str(con) + 'minute_' + str(i) + '.png' )
    plt.close()
    
    
    