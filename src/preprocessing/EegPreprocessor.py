import os, sys
import numpy as np
import mne
import EegPreprocessor as preprocessor


#Preprocessing the file obtained from filepath
DEFAULT_EOG = ['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']

#Channel position function for Biosemi machine: Data fromat EDF, BDF, Brainvision
def load_biosemi_montage(string = 'biosemi64'):
    montage = mne.channels.make_standard_montage(string)
    return montage


#Loading the raw data
def load_raw_data(filepath, montage = None, eog = DEFAULT_EOG, preload = False):
    fname = os.path.abspath(filepath)
    ext = os.path.splitext(fname)[1][1:].lower()
    
    if ext =='bdf':
        montage = load_biosemi_montage()
        raw = mne.io.read_raw_bdf(filepath, montage, eog, preload =preload)
        
    elif ext =='edf':
        montage = load_biosemi_montage()
        raw = mne.io.read_raw_edf(filepath, montage, eog, preload =preload)
    
    elif ext == 'vhdr':
        eog = ('HEOGL', 'HEOGR', 'VEOGb')
        montage = 'C:\\Users\\J_CHOWD\\Desktop\\Project_CAE\\Deliver Microstates\\Channel_location_file\\Cap63.locs'
        raw = mne.io.read_raw_brainvision(vhdr_fname = fname, eog = eog, misc= 'auto', scale= 1., 
                                          preload = False, verbose = False)
        dig_montage = mne.channels.read_custom_montage(fname = montage)
        raw.set_montage(dig_montage, verbose= False, raise_if_subset = False)
 
    return raw


def preprocess_raw_data(filepath, low_freq = 0.1, high_freq = 100):
    raw = load_raw_data(filepath, montage = None, preload = True)
    raw.plot(duration = 0.2)
    
#Band pass filtering of data
    raw.load_data()
    raw.filter(low_freq, high_freq, fir_design='firwin')
    raw.plot(duration = 0.2)
#Notch fitering to remove the line noise
    raw.notch_filter(np.arange(60, 241, 60), fir_design='firwin')
    
#Setting the average EEG reference
    raw.set_eeg_reference('average', projection=False)
    
    return raw


    


