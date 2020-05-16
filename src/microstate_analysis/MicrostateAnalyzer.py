import numpy as np
import ModifiedKmeans as kmeans
import MicrostateMapsAnalysis
import microstates


def analyzeMicrostate(raw, channels, n_maps = 4, n_runs = 10, maxerr = 1e-6, maxiter = 1000, 
                      doplot = False):
    data = raw.pick(picks = channels).get_data().T
    maps, labels, gfp_peaks, gev, cv = kmeans(data, n_maps, n_runs, maxerr, maxiter, doplot)
    
    return maps, labels, gev, cv


def analyze_micrsotate(raw,channels):
    #Setting the Bio Semi 64 channel montage
    raw.set_montage('biosemi64')

    #Selecting the EEG channels only
    raw.pick_types(meg=False, eeg=True)
    
    # Formation of raw instance with primary channels AF7, AF8, FT7, FT8 
    
    data = raw.pick_channels(ch_names = channels).get_data()
    
    # Segment the data into 4 microstates
    maps, segmentation = microstates.segment(data, n_states= 4, max_n_peaks=10000000000, max_iter=5000, 
                                             normalize=True)
    
    return maps, segmentation
   
def analayzeMicrsotateResidue(raw):
    #Setting the Bio Semi 64 channel montage
    raw.set_montage('biosemi64')

    #Selecting the EEG channels only
    raw.pick_types(meg = False, eeg = True, eog = False, stim = False)
    
    # Formation of the residue raw instance/object
    data = raw.drop_channels(ch_names = ['AF7', 'AF8', 'FT7', 'FT8', 'Fp1', 'AF3', 'F3', 'F5', 'F7',
                                        'Fp2', 'AF4', 'F4', 'F6', 'F8', 'F7', 'F5', 'FC5', 'C5', 'T7',
                                        'F8', 'F6', 'FC6', 'C6', 'T8'] ).get_data()

    mapsResidue, segmentationResidue = microstates.segment(data, n_states=4,
                                                             max_n_peaks=10000000000, 
                                                             max_iter=5000, normalize=True)
    return mapsResidue, segmentationResidue


    
    