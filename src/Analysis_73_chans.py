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
import argparse, os, sys, time


def read_edf(filename):
    """Basic EDF file format reader

    EDF specifications: http://www.edfplus.info/specs/edf.html

    Args:
        filename: full path to the '.edf' file
    Returns:
        chs: list of channel names
        fs: sampling frequency in [Hz]
        data: EEG data as numpy.array (samples x channels)
    """

    def readn(n):
        """read n bytes."""
        return np.fromfile(fp, sep='', dtype=np.int8, count=n)

    def bytestr(bytes, i):
        """convert byte array to string."""
        return np.array([bytes[k] for k in range(i*8, (i+1)*8)]).tostring()

    fp = open(filename, 'r')
    x = np.fromfile(fp, sep='', dtype=np.uint8, count=256).tostring()
    header = {}
    header['version'] = x[0:8]
    header['patientID'] = x[8:88]
    header['recordingID'] = x[88:168]
    header['startdate'] = x[168:176]
    header['starttime'] = x[176:184]
    header['length'] = int(x[184:192]) # header length (bytes)
    header['reserved'] = x[192:236]
    header['records'] = int(x[236:244]) # number of records
    header['duration'] = float(x[244:252]) # duration of each record [sec]
    header['channels'] = int(x[252:256]) # ns - number of signals
    n_ch = header['channels']  # number of EEG channels
    header['channelname'] = (readn(16*n_ch)).tostring()
    header['transducer'] = (readn(80*n_ch)).tostring().split()
    header['physdime'] = (readn(8*n_ch)).tostring().split()
    header['physmin'] = []
    b = readn(8*n_ch)
    for i in range(n_ch): header['physmin'].append(float(bytestr(b, i)))
    header['physmax'] = []
    b = readn(8*n_ch)
    for i in range(n_ch): header['physmax'].append(float(bytestr(b, i)))
    header['digimin'] = []
    b = readn(8*n_ch)
    for i in range(n_ch): header['digimin'].append(int(bytestr(b, i)))
    header['digimax'] = []
    b = readn(8*n_ch)
    for i in range(n_ch): header['digimax'].append(int(bytestr(b, i)))
    header['prefilt'] = (readn(80*n_ch)).tostring().split()
    header['samples_per_record'] = []
    b = readn(8*n_ch)
    for i in range(n_ch): header['samples_per_record'].append(float(bytestr(b, i)))
    nr = header['records']
    n_per_rec = int(header['samples_per_record'][0])
    n_total = int(nr*n_per_rec*n_ch)
    fp.seek(header['length'],os.SEEK_SET)  # header end = data start
    data = np.fromfile(fp, sep='', dtype=np.int16, count=n_total)  # count=-1
    fp.close()

    # re-order
    data = np.reshape(data,(n_per_rec,n_ch,nr),order='F')
    data = np.transpose(data,(0,2,1))
    data = np.reshape(data,(n_per_rec*nr,n_ch),order='F')

    # convert to physical dimensions
    for k in range(data.shape[1]):
        d_min = float(header['digimin'][k])
        d_max = float(header['digimax'][k])
        p_min = float(header['physmin'][k])
        p_max = float(header['physmax'][k])
        if ((d_max-d_min) > 0):
            data[:,k] = p_min+(data[:,k]-d_min)/(d_max-d_min)*(p_max-p_min)

    print(header)
    return header['channelname'].split(),\
           header['samples_per_record'][0]/header['duration'],\
           data
channelData = read_edf('C:/projects/eeg_microstates/src/test.edf')
data = channelData[2]
n_maps = 4
maps, L_, gfp_peaks, gev,cv = testing.kmeans(data, n_maps, n_runs = 5,maxerr = 1e-6, maxiter = 200 )


#for i in range(1):
#    print(__doc__)
    
    
#    DEFAULT_EOG = ['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']
    
#    def get_raw_data_file_path():
#        #root = tk.Tk()
#        #root.withdraw()
#        #data_file_path  = filedialog.askopenfilename()
#        return 'C:/projects/eeg_microstates/src/test.edf'
    
#    #Channel position function for Biosemi machine
#    def load_biosemi_montage(string = 'biosemi64'):
#        montage = mne.channels.make_standard_montage(string)
#        return montage
    
#    #Loading the raw data
#    def load_raw_data(filepath, montage, eog = DEFAULT_EOG):
#        raw = mne.io.read_raw_edf(filepath, montage, eog, preload =True)
#        return raw
    
    
#    def preprocess_raw_data():
#        print("Please enter the raw data file")
#        filepath = get_raw_data_file_path()
        
#        montage = load_biosemi_montage()
#        raw = load_raw_data(filepath, montage)
        
#    #Using the average EEG reference
#        raw.set_eeg_reference('average')
#    #High pass filtering of data
#        raw.filter(0.2, None)
#        return raw
#    raw = preprocess_raw_data()
    
#    data = raw.get_data()
#    #data = np.resize(data,(30,))

    
    
    
    
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
    
    
    
    
    
    
    