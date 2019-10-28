import tkinter as tk
from tkinter import filedialog


import numpy as np
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
    
    #raw.plot_psd(tmax=np.inf, fmax=500)
    #preprocessor.resample_raw_data(raw)
    
#EOG
#    average_eog = preprocessor.get_average_eog(raw)
#    
#    plotter.print_average_eog(average_eog)
#    ica, eog_inds, scores, eog_average, eog_epochs, raw_copy = preprocessor.apply_ICA(raw)
#    print(ica)
#    #plotter.print_ICA(ica, raw, eog_average, eog_inds, scores, eog_epochs, raw_copy)
    return raw
raw = preprocess_raw_data()

for i in range(64,72):
    event_id = i
    epochs, data = preprocessor.find_eog_artifacts(raw, event_id)
    plotter.plot_EOG_artifacts(epochs, data)
