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
    tmin, tmax = 0, 200 # Using first 200 seconds of data
    raw = raw.crop(tmin, tmax).load_data()# Setup for raw data saving memory by cropping the raw data before loading it
    fmin, fmax = 45, 70 # Looking at the feequencies between 45-70 Hz
    n_fft = 2048 # The FFt size. Ideally a power of 2
    selection = ['AF7','AF8','FT7','FT8']
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False,
                       stim = False, selection=selection)
    raw.plot_psd(area_mode='range', tmax=200.0, picks=picks, average=False)

#**Removing power-line noise with notch filtering. This Removing can be done with a Notch filter, directly on the Raw object, specifying an array of frequency to be cut off:

    raw.notch_filter(np.arange(60, 241, 60), picks=picks, fir_design='firwin')
    raw.plot_psd(area_mode='range', tmax=200.0, picks=picks, average = False)




***********
Removing power-line noise with low-pass filtering
Since EEG signal bands like alpha, beta gamma delta are of low frequencies, below the peaks of power-line noise we can simply low pass filter the data.

# low pass filtering below 50 Hz
    raw.filter(None, 50., fir_design='firwin')
    raw.plot_psd(area_mode='range', tmax=200.0, picks=picks, average=False)

***High-pass filtering to remove slow drifts


    raw.filter(1., None, fir_design='firwin')
    raw.plot_psd(area_mode='range', tmax=200.0, picks=picks, average=False)
    # Band pass filtering of data
    raw.filter(45, 70, fir_design = 'firwin')
    
    # Notch fitering to remove the line noise
    #raw.notch_filter(np.arange(60, 241, 60), fir_design)
    
    # Setting the average EEG reference
    raw.set_eeg_reference('average', projection = False)
    
    return raw


    


