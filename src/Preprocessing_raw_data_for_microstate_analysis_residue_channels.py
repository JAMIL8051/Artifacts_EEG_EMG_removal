import tkinter as tk
from tkinter import filedialog
import numpy as np
import mne
import EegPreprocessor as preprocessor
#import eeg_visualizer as plotter
import microstates
import MicrostateAnalyzer as ms_analyze

#import EmpiricalModeDecomposition as emd
#import pca_ica_gfp_freq_bands as analysis gfp_analysis
#import mara as mara1
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
    print("Please give a start time in seconds for preprocessing of your data")
    tmin = int(input())
    print("Please give an end time in seconds for preprocessing of your data")
    tmax = int(input())

    #raw.plot_psd(tmax = np.inf, fmax=512)

# Resampling using pyprep package
    nd, bads, channel_correlations, high_freq_noise_per_channel = preprocessor.resample_raw_data(raw, tmin, tmax)
    return raw, bads    
raw, bads = preprocess_raw_data()

maps_residue, raw_residue = ms_analyze.residue_analysis(raw,bads,tmin=0,tmax=50)
microstates.plot_maps(maps_residue, info = raw_residue.info)