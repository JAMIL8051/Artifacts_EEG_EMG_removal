import tkinter as tk
from tkinter import filedialog
import numpy as np
import mne
import EegPreprocessor as preprocessor
#import eeg_visualizer as plotter
#import microstates
#import MicrostateAnalyzer as ms_analyze
import scratch as testing
#import EmpiricalModeDecomposition as emd
#import pca_ica_gfp_freq_bands as analysis gfp_analysis
#import mara as mara1
#from itertools import combinations
    

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
    
    
def preprocess_raw_data():
    print("Please enter the raw data file")
    filepath = get_raw_data_file_path()
        
    montage = load_biosemi_montage()
    raw = load_raw_data(filepath, montage)
        
#Using the average EEG reference
    raw.set_eeg_reference('average')
#High pass filtering of data
    raw.filter(0.1, None)
    
    
    raw.plot_psd(tmax = np.inf, fmax=512)
    return raw   
# Resampling using pyprep package
raw = preprocess_raw_data()
raw = raw.resample(sfreq = 256,npad='auto')
data = raw.get_data()

preprocessor.apply_ICA(raw)


#raw_pick_bad_channels = raw_copy1.pick_channels(ch_names = ['Fp1'])

##EEG microstates analysis on bad_data

data1 = np.resize(data1,(30,19200))
#data = data.astype(np.float16)


#data_bad_channels_tr = data_bad_channels.transpose()
#data_bad_channels = np.resize(data_bad_channels,(len(bad_channels),len(data_bad_channels_tr)))

print("Ready for EEG microstate analysis for bad channels from PyPrep")
n_maps = 4
print('Applying the algorithm')
maps = testing.kmeans(data1, n_maps, n_runs = 5,maxerr = 1e-6, maxiter = 500)






        



