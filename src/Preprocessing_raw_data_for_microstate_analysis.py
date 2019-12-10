import tkinter as tk
from tkinter import filedialog
import numpy as np
import mne
import EegPreprocessor as preprocessor
#import eeg_visualizer as plotter
import microstates
#import MicrostateAnalyzer as ms_analyze

#import EmpiricalModeDecomposition as emd
#import pca_ica_gfp_freq_bands as analysis gfp_analysis
#import mara as mara1
from itertools import combinations, permutations
    
for i in range(2):
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
    
        #raw.plot_psd(tmax = np.inf, fmax=500)
    
    # Resampling using pyprep package
        nd, bads, channel_correlations, high_freq_noise_per_channel = preprocessor.resample_raw_data(raw, tmin, tmax)

##EOG detection phase
#    average_eog = preprocessor.get_average_eog(raw)
#    plotter.print_average_eog(average_eog)
#    for i in range(64,67):
#        event_id = i
#        epochs, data = preprocessor.find_eog_artifacts(raw, event_id)
#        plotter.plot_EOG_artifacts(epochs, data)
    
####Artifact correction with SSP
#    projs, eog_projs = preprocessor.compute_SSP_projection_eog(raw)
#    raw_SSP = preprocessor.apply_SSP_projections(raw, eog_projs)
#    epochs_no_proj, epochs_proj, evoked = preprocessor.demonstrate_SSP_cleaning(raw_SSP)
#    plotter.get_projections(projs, raw_SSP, eog_projs)
#    fig = plotter.plot_demonstration_SSP_cleaning(epochs_no_proj, epochs_proj, evoked)
#    
#Application of ICA:
#    ica, eog_inds, scores, eog_average, eog_epochs, raw_copy = preprocessor.apply_ICA(raw)
#    print(ica)
#    plotter.print_ICA(ica, raw, eog_average, eog_inds, scores, eog_epochs, raw_copy)
#    return nd, bads, high_freq_noise_per_channel, channel_correlations, evoked 
        return raw, bads, nd    
    raw, bads, nd = preprocess_raw_data()
        

#raw = preprocess_raw_data()

    raw_copy1 = raw.copy()
    raw_copy2 = raw.copy()


## Formation of raw instance with bad_channels on the basis of Pyprep 
    bad_channels = bads
    raw_pick_bad_channels = raw_copy1.pick_channels(ch_names = bad_channels)

##EEG microstates analysis on bad_data
    data_bad_channels = raw_pick_bad_channels.get_data()
    data_bad_channels_tr = data_bad_channels.transpose()
    data_bad_channels = np.resize(data_bad_channels,(len(bad_channels),len(data_bad_channels_tr)))

    print("EEG microstate analysis for bad channels from PyPrep")

    for i in range(2):
        n_states_bad_channel = int(input("Please provide the number of Microstates: "))
        if n_states_bad_channel <2 :
            print("The number of microstates must be equal greater than or equal to 2" )
        n_inits_bad_channel = int(input("Please give the number of random initializations to use for the k-means algorithm: "))
        maps_bad_channels, segmentation_bad_channels = microstates.segment(data_bad_channels, n_states= n_states_bad_channel, n_inits = n_inits_bad_channel)
        microstates.plot_maps(maps_bad_channels, raw_pick_bad_channels.info)
    
##Formation of the residue raw instance/object
    raw_residue= raw_copy2.drop_channels(ch_names = bad_channels)
    
##Segementation of raw_residue to chunks
    residue_channels = raw_residue.ch_names
    perm_residue_channel = combinations(residue_channels, len(bads))
    perm_residue_channels = list(perm_residue_channel)
    for i in range(int(len(perm_residue_channels)/32)):
        raw_chunks = raw_residue.copy()
        raw_chunk = raw_chunks.pick_channels(ch_names = perm_residue_channels[i]) 

##EEG microstates analysis
        data_residue = raw_chunk.get_data()
        data_residue_tr = data_residue.transpose()
        data_residue = np.resize(data_residue,(len(bads), len(data_residue_tr)))
        
        print("EEG microstate analysis for rest of the channels from PyPrep")
    
        for i in range(1):
#            n_states_residue = int(input("Please provide the number of Microstates: "))
#            if n_states_residue <2 :
#               print("The number of microstates must be equal greater than or equal to 2" )
#            n_inits_residue = int(input("Please give the number of random initializations to use for the k-means algorithm: "))
            maps_residue, segmentation_residue = microstates.segment(data_residue, n_states= 4, n_inits = 300)
            microstates.plot_maps(maps_residue, raw_chunk.info)


#data1 = raw.get_data()
#data = np.resize(data1,(64,55296))


##EEG Microstates Analysis
## Segment the data in number of microstates
#
##events = mne.find_events(raw)
#
#maps, segmentation = microstates.segment(data, n_states= n_states, n_inits = n_inits)
#
## Plot the topographic maps of the microstates and the segmentation
#print(" Visualizing the topographical maps of the EEG Micrsotates ")
#
#microstates.plot_maps(maps, raw.info)
##Plotting the segementation for first 600 time samples
#microstates.plot_segmentation(segmentation[:600], raw.get_data()[:, :600],
#                              raw.times[:600])
#omega_signal, f_inst_signal, hilbert_signal = emd.apply_EMD(raw)
#pca_transformed_data = mara1.mara(raw)
 #gfp_analysi.pca_ica_gfp_freq_bands(raw)

