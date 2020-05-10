import numpy as np
import microstates_k_means
import microstates



def analyze_micrsotate(raw):
    #Setting the Bio Semi 64 channel montage
    raw.set_montage('biosemi64')

    #Selecting the EEG channels only
    raw.pick_types(meg=False, eeg=True)
    
    # Formation of raw instance with primary channels AF7, AF8, FT7, FT8 
    primary_raw = raw.copy()
    primary_raw = priamry_raw.pick_channels(ch_names = ['AF7','AF8','FT7','FT8'])
    
    # Segment the data into 4 microstates
    priamry_maps, priamry_segmentation = microstates.segment(primary_raw.get_data(), n_states= 4, 
                                                             max_n_peaks=10000000000, 
                                                             max_iter=5000, normalize=True)
    
    # Formation of raw instance with channels adajecnet to AF7. 
    AF7_raw = raw.copy()
    AF7_raw = AF7_raw.pick_channels(ch_names =['Fp1', 'AF3', 'F3', 'F5', 'F7'])
    
    # Segment the data into 4 microstates
    AF7_maps, AF7_segmentation = microstates.segment(AF7_raw.get_data(), n_states= 4, 
                                                             max_n_peaks=10000000000, 
                                                             max_iter=5000, normalize=True)
    
    # Formation of raw instance with channels adajecnet to AF8. 
    AF8_raw = raw.copy()
    AF8_raw = AF8_raw.pick_channels(ch_names =['Fp2', 'AF4', 'F4', 'F6', 'F8'])
    
    # Segment the data into 4 microstates
    AF8_maps, AF8_segmentation = microstates.segment(AF8_raw.get_data(), n_states= 4, 
                                                             max_n_peaks=10000000000, 
                                                             max_iter=5000, normalize=True)

    # Formation of raw instance with channels adajecnet to FT7. 
    FT7_raw = raw.copy()
    FT7_raw = FT7_raw.pick_channels(ch_names =['F7', 'F5', 'FC5', 'C5', 'T7'])
    
    # Segment the data into 4 microstates
    FT7_maps, FT7_segmentation = microstates.segment(FT7_raw.get_data(), n_states= 4, 
                                                             max_n_peaks=10000000000, 
                                                             max_iter=5000, normalize=True)

    # Formation of raw instance with channels adajecnet to FT8. 
    FT8_raw = raw.copy()
    FT8_raw = FT8_raw.pick_channels(ch_names =['F8', 'F6', 'FC6', 'C6', 'T8'])
    
    # Segment the data into 4 microstates
    FT8_maps, FT8_segmentation = microstates.segment(FT8_raw.get_data(), n_states= 4, 
                                                             max_n_peaks=10000000000, 
                                                             max_iter=5000, normalize=True)
    

    return primary_maps, AF7_maps, AF8_maps, FT7_maps, FT8_maps
   
def analayze_micrsotate_residue(raw):
    #Setting the Bio Semi 64 channel montage
    raw.set_montage('biosemi64')

    #Selecting the EEG channels only
    raw.pick_types(meg=False, eeg=True,eog=False, stim = False)
    
    # Formation of the residue raw instance/object
    raw_residue = raw.drop_channels(ch_names = ['AF7','AF8','FT7','FT8',
                                                'Fp1', 'AF3', 'F3', 'F5', 'F7',
                                                'Fp2', 'AF4', 'F4', 'F6', 'F8',
                                                'F7', 'F5', 'FC5', 'C5', 'T7',
                                                'F8', 'F6', 'FC6', 'C6', 'T8'] )
  
    
    
    maps_residue, segmentation_residue = microstates.segment(raw_residue.get_data(), n_states=4,
                                                             max_n_peaks=10000000000, 
                                                             max_iter=5000, normalize=True)
    return maps_residue


    
    