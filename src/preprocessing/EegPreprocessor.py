import mne



#Preprocessing the file obtained from filepath
print(__doc__)


DEFAULT_EOG = ['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']

#Channel position function for Biosemi machine
def loadBiosemiMontage(string = 'biosemi64'):
    montage = mne.channels.make_standard_montage(string)
    return montage


#Loading the raw data
def load_raw_data(filepath, montage, eog = DEFAULT_EOG):
    raw = mne.io.read_raw_bdf(filepath, montage, eog, preload =True)
    return raw


def preprocessRawData(filepath, low_freq = 0.1, high_freq = 100, fir_design = 'firwin'):
    
    raw = load_raw_data(filepath)        
    
    # Band pass filtering of data
    raw.filter(low_freq, high_freq, fir_design = 'firwin')
    
    # Notch fitering to remove the line noise
    raw.notch_filter(np.arange(60, 241, 60), fir_design)
    
    # Setting the average EEG reference
    raw.set_eeg_reference('average', projection = False)
    
    return raw


    


