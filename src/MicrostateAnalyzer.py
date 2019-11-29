import numpy as np
import microstates


def bad_channel_analysis(raw, bads):
    # Formation of raw instance with bad_channels on the basis of Pyprep 
    bad_channels = bads
    raw_pick_bad_channels = raw.pick_channels(ch_names = bad_channels)

#EEG microstates analysis on bad_data
    data_bad_channels = raw_pick_bad_channels.get_data()
    data_bad_channels = np.resize(data_bad_channels,(len(raw_pick_bad_channels.ch_names), 10800))
    
    n_states_bad_channel = int(input("Please provide the number of Microstates: "))
    if n_states_bad_channel <2 :
        print("The number of microstates must be equal greater than or equal to 2" )
    n_inits_bad_channel = int(input("Please give the number of random initializations to use for the k-means algorithm: "))
    maps_bad_channels, segmentation_bad_channels = microstates.segment(data_bad_channels, n_states= n_states_bad_channel, n_inits = n_inits_bad_channel)
    return maps_bad_channels, raw_pick_bad_channels
    #microstates.plot_maps(maps_bad_channels, raw_pick_bad_channels.info)

def residue_analysis(raw, bads, tmin,tmax):
    bad_channels = bads
    # Formation of the residue raw instance/object
    raw_residue = raw.drop_channels(ch_names = bad_channels)
    data_residue = raw_residue.crop(tmin,tmax).load_data()
    # EEG microstates analysis
    data_residue = data_residue.get_data()
    data_residue = np.resize(data_residue, (62, 10800))

    n_states_residue = int(input("Please provide the number of Microstates: "))
    if n_states_residue < 2:
        print("The number of microstates must be equal greater than or equal to 2")

    n_inits_residue = int(
        input("Please give the number of random initializations to use for the k-means algorithm: "))
    maps_residue, segmentation_residue = microstates.segment(data_residue, n_states=n_states_residue,
                                                             n_inits=n_inits_residue)
    return maps_residue, raw_residue


    
    #microstates.plot_maps(maps_residue, raw_residue.info)