import tkinter as tk
from tkinter import filedialog
#import os
#from pathlib import Path
import mne
from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs
import numpy as np
from pyprep.noisy import Noisydata
import EegPreprocessor as preprocessor
import eeg_visualizer as plotter
print(__doc__)

def get_raw_data_file_path():
    root = tk.Tk()
    root.withdraw()

    data_file_path  = filedialog.askopenfilename()
    return data_file_path

def preprocess_raw_data():
    print("Please enter the raw data file")
    filepath = get_raw_data_file_path()
    montage = preprocessor.load_biosemi_montage()
    raw = preprocessor.load_raw_data(filepath, montage)
    
    raw.plot_psd(tmax=np.inf, fmax=500)
    #preprocessor.resample_raw_data(raw)
    average_eog = preprocessor.get_average_eog(raw)
    plotter.print_average_eog(average_eog)
    ica = preprocessor.apply_ICA(raw)
    print(ica)
    plotter.print_ICA(ica,raw)

preprocess_raw_data()