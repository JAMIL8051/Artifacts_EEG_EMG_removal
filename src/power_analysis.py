from tkinter import filedialog
import tkinter as tk
import numpy as np
import mne
from 2_preprocessing import EegPreprocessor as preprocessor
import matplotlib.pyplot as plt

#Preprocessing file
print(__doc__)


DEFAULT_EOG = ['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']


def get_raw_data_file_path():
    root = tk.Tk()
    root.withdraw()
    data_file_path  = filedialog.askopenfilename()
    return data_file_path


#Channel position function for Biosemi machine
def load_biosemi_montage(string = 'biosemi64'):
    montage = mne.channels.make_standard_montage(string)
    return montage


#Loading the raw data
def load_raw_data(filepath, montage, eog = DEFAULT_EOG):
    raw = mne.io.read_raw_bdf(filepath, montage, eog, preload =True)
    return raw

#Preprocessing the raw data
def preprocess_raw_data():
    print("Please enter the raw data file")
    filepath = get_raw_data_file_path()    
    montage = load_biosemi_montage()
    raw = load_raw_data(filepath, montage)        
    
#High pass filtering of data
    raw.filter(0.1, 100.0, fir_design='firwin')
    
#Notch fitering to remove the line noise
    raw.notch_filter(np.arange(60, 241, 60), fir_design='firwin')
    
#Setting the average EEG reference
    raw.set_eeg_reference('average', projection=False)
    
    return raw


# Function for Power_analysis
def final_band_power(raw,channel,N,fs,tmin=None, tmax=None,epoch_time=2):
    """
    raw = Mne.raw object
    picks = no. of channels to include, type must be string
    N = No. of samples for FFT algorithm
    fs = sampling frequecy 
    tmin = starting time of the trials
    tmax = end time of the trials
    epoch_time = duration of the epoch by default 2 seconds
    """
    raw = raw.pick(picks=channel)
    
    if (tmin==None and tmax == None):
        n_epochs = 100
    else:
        n_epochs = int((tmax-tmin)/epoch_time)
    
    data = raw.get_data()
    data = data/1e-06
    #Normalizing the data before power calculations
    data = (data-data.mean())/(data.std())
    n_channels = len(raw.ch_names)
    epoch_power_45_70Hz = np.zeros((n_channels,n_epochs,1),dtype = 'float')
    
    for chan in range(n_channels):
        for epoch in range(n_epochs):
            # A small investigation leads to the fact that I had to remove the first sample of 
            # from evey channel as the value was very very close zero.  
            epoch_data = data[chan, ((N*epoch)+1):(N*(epoch+1)+1)] #Formation of 2s epoch data with N samples
            epoch_data = np.fft.fft(epoch_data, N)
            #calculation of k in the formula
            k_lower = int(45*(N/fs)) # Forcing these to integers otherwise index error problem will pop up 
            k_upper = int(70*(N/fs))
            temp = np.zeros((50),dtype='float')
            for k in range(k_lower, k_upper):
                val = (np.abs(epoch_data[k_lower])) ** 2 + (np.abs(epoch_data[N - k_lower])) ** 2
                temp[k-90] = val
            
            power_45_70Hz = (1/N**2)*sum(temp)
            epoch_power_45_70Hz[chan,epoch] = power_45_70Hz
            
    return epoch_power_45_70Hz


#Main code start here
raw = preprocess_raw_data()

#Duplicating the raw object: Util function??
primary_raw = raw.copy()
primary_picks = ['AF7','AF8','FT7','FT8']
primary_power_45_70Hz = final_band_power(primary_raw, channel = primary_picks, N=1024,fs=512)
#primary_raw = primary_raw.pick(picks = primary_picks)

AF7_raw = raw.copy()
AF7_picks = ['Fp1', 'AF3', 'F3', 'F5', 'F7']
AF7_power_45_70Hz = final_band_power(AF7_raw, channel = AF7_picks, N=1024,fs=512)
#AF7_raw = AF7_raw.pick(picks = AF7_picks)

AF8_raw = raw.copy()
AF8_picks = ['Fp2', 'AF4', 'F4', 'F6', 'F8']
AF8_power_45_70Hz = final_band_power(AF8_raw, channel = AF8_picks, N=1024,fs=512)
#AF8_raw = AF8_raw.pick(picks = AF8_picks)

FT7_raw = raw.copy()
FT7_picks = ['F7', 'F5', 'FC5', 'C5', 'T7'] 
FT8_power_45_70Hz = final_band_power(FT7_raw, channel = FT7_picks, N=1024,fs=512)
#FT7_raw = FT7_raw.pick(picks = FT7_picks)

FT8_raw = raw.copy()
FT8_picks = ['F8', 'F6', 'FC6', 'C6', 'T8'] 
FT8_power_45_70Hz = final_band_power(FT8_raw, channel = FT8_picks, N=1024,fs=512)

#FT8_raw = FT8_raw.pick(picks = FT8_picks)
Fz_raw = raw.copy()
Fz_picks = ['Fz'] 
Fz_power_45_70Hz = final_band_power(Fz_raw, channel = Fz_picks, N=1024,fs=512)

#Detection-phase of EMG contaminated segments. detection phase file??

# Collection of the data matrices for numerical calculation of power based  on FFT algorithm??
primary_data = primary_raw.get_data()

#Data of the 5 adjacent channels of the channel: AF7
AF7_data = AF7_raw.get_data()
#Data of the 5 adjacent channels of the channel: AF8
AF8_data = AF8_raw.get_data()
#Data of the 5 adjacent channels of the channel: FT7
FT7_data = FT7_raw.get_data()
#Data of the 5 adjacent channels of the channel: FT8
FT8_data = FT8_raw.get_data()

#Data of the central "Fz" channel of the primary channels: AF&, AF8, FT7, FT8
Fz_data = Fz_raw.get_data()

# Combining the data excluding the Fz channel-> not sure which script to export!!!
combined_data = np.concatenate((primary_data,AF7_data,AF8_data,FT7_data,FT8_data),axis = 0)


#Step 1 :Finding the threshold val without simulated data.

threshold_val_AF7 = primary_power_45_70Hz[0,1:].mean(dtype='float')
threshold_val_AF8 = primary_power_45_70Hz[1,1:].mean(dtype='float')
threshold_val_FT7 = primary_power_45_70Hz[2,1:].mean(dtype='float')
threshold_val_FT8 = primary_power_45_70Hz[3,1:].mean(dtype='float')
 
#threshold_val_AF7 = primary_power_45_70Hz[0,1:].mean(dtype='float')+ 1*primary_power_45_70Hz[0,1:].std(dtype='float')
#threshold_val_AF8 = primary_power_45_70Hz[1,1:].mean(dtype='float')+ 1*primary_power_45_70Hz[1,1:].std(dtype='float')
#threshold_val_FT7 = primary_power_45_70Hz[2,1:].mean(dtype='float')+ 1*primary_power_45_70Hz[2,1:].std(dtype='float')
#threshold_val_FT8 = primary_power_45_70Hz[3,1:].mean(dtype='float')+ 1*primary_power_45_70Hz[3,1:].std(dtype='float')
threshold_val = [threshold_val_AF7, threshold_val_AF8, threshold_val_FT7, threshold_val_FT8]


#Using simulated data: Will do later


#Step 2 :Finding the channel epoch indices which are greater than threshold
chan_epoch_indice_primary = np.zeros((len(primary_raw.ch_names),100),dtype='int')

for chan in range(len(primary_raw.ch_names)):
    for epoch in range(50):
        if primary_power_45_70Hz[chan,epoch] > threshold_val:
            chan_epoch_indice_primary[chan,epoch] = epoch 

chan_epoch_indices_AF7 = np.zeros((len(AF7_raw.ch_names),50),dtype='int')
chan_epoch_indices_AF8 = np.zeros((len(AF8_raw.ch_names),50),dtype='int')
chan_epoch_indices_FT7 = np.zeros((len(FT7_raw.ch_names),50),dtype='int')
chan_epoch_indices_FT8 = np.zeros((len(FT8_raw.ch_names),50),dtype='int')


for idx in chan_epoch_indice_primary:
    if  chan_epoch_indice_primary.any() == 0:
        for chan in range(len(AF7_raw.ch_names)):
            for epoch in range(50):
                if AF7_power_45_70Hz[chan,epoch] >threshold_val[0]:
                    chan_epoch_indices_AF7[chan,epoch] = epoch
    elif chan_epoch_indice_primary.any() == 1:
        for chan in range(len(AF8_raw.ch_names)):
            for epoch in range(50):
                if AF8_power_45_70Hz[chan,epoch] >threshold_val[1]:
                    chan_epoch_indices_AF8[chan,epoch] = epoch
    elif chan_epoch_indice_primary.any() == 2:
        for chan in range(len(FT7_raw.ch_names)):
            for epoch in range(50):
                if AF8_power_45_70Hz[chan,epoch] >threshold_val[2]:
                    chan_epoch_indices_FT7[chan,epoch] = epoch
    elif chan_epoch_indice_primary.any() == 3:
        for chan in range(len(FT7_raw.ch_names)):
            for epoch in range(50):
                if AF8_power_45_70Hz[chan,epoch] >threshold_val[3]:
                    chan_epoch_indices_FT7[chan,epoch] = epoch
print('thres_mara')
    
                
#Step 2a: Decomposing the power data array in to epochs by power



#Step 3: Locating the channels



#Step 3a: Decomposing the power data array into channels by epochs




#Step 4: Locating the contaminated epochs and finding the corresponding channel indices 
#for shuffling the contaminated data
