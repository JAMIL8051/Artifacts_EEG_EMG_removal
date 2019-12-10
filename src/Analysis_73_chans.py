import tkinter as tk
from tkinter import filedialog
import numpy as np
import mne
#import EegPreprocessor as preprocessor
#import eeg_visualizer as plotter
import microstates
#import MicrostateAnalyzer as ms_analyze

#import EmpiricalModeDecomposition as emd
#import pca_ica_gfp_freq_bands as analysis gfp_analysis
#import mara as mara1
import scratch as testing

    
for i in range(1):
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
        raw.filter(0.2, None)
        return raw
    raw = preprocess_raw_data()
    
    data = raw.get_data()
    #data = np.resize(data,(30,))
    n_maps = 4
    #maps, L_, gfp_peaks, gev,cv = testing.kmeans(data, n_maps, n_runs = 5,maxerr = 1e-6, maxiter = 200 )
    
    
    
    
#    bad_channels_1st_32 = ['FT7', 'FC5', 'P9', 'FC3', 'POz', 'PO3', 'Iz'] 
#    bad_channels_2nd_32 = ['Fp2', 'FT8', 'C4', 'AF8', 'F8']
#    
#    raw_copy1 = raw.copy()
#    raw_copy2 = raw.copy()
#    
#    
#
#    raw_pick_bad_channels_1st_32 = raw_copy1.pick_channels(ch_names = bad_channels_1st_32)
#    raw_pick_bad_channels_2nd_32 = raw_copy2.pick_channels(ch_names = bad_channels_2nd_32)
#    
#    data_bad_channels = raw_pick_bad_channels_1st_32.get_data()
#    data_bad_channels_tr = data_bad_channels.transpose()
#    data_bad_channels = np.resize(data_bad_channels,(len(bad_channels_1st_32),len(data_bad_channels_tr)))
#    
#    data_bad_channels_2 = raw_pick_bad_channels_2nd_32.get_data()
#    data_bad_channels_tr_2 = data_bad_channels_2.transpose()
#    data_bad_channels_2 = np.resize(data_bad_channels_2,(len(bad_channels_2nd_32),len(data_bad_channels_tr_2)))
#    
#    maps_bad_channels_1st_32, segmentation_bad_channels_1st_32 = microstates.segment(data_bad_channels, n_states= 4, n_inits = 300)
#    microstates.plot_maps(maps_bad_channels_1st_32, raw_pick_bad_channels_1st_32.info)
#    
#    maps_bad_channels_2nd_32, segmentation_bad_channels_2nd_32 = microstates.segment(data_bad_channels_2, n_states= 4, n_inits = 300)
#    microstates.plot_maps(maps_bad_channels_2nd_32, raw_pick_bad_channels_2nd_32.info)
    
    
    
    
    
    
    