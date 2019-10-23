import mne
import numpy as np
from pyprep.noisy import Noisydata
from mne.preprocessing import create_eog_epochs
from mne.preprocessing import ICA
BIOSEMI_MACHINE_NAME = 'biosemi'
DEFAULT_MONTAGE_MACHINE = BIOSEMI_MACHINE_NAME + '64'
DEFAULT_EOG = ['EXG1','EXG2','EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']

def load_montage(channel_names,
                 channel_file_path, 
                 kind = DEFAULT_MONTAGE_MACHINE,
                 unit ='m',
                 transform = False):
    montage = mne.channels.read_montage(kind, channel_names, channel_file_path, unit, transform)
    return montage

def load_biosemi_montage(biosemi_channel_count = 64):
    montage = load_montage(kind = BIOSEMI_MACHINE_NAME + str(BIOSEMI_MACHINE_NAME))
    return montage

def load_raw_data(filepath, montage, eog = DEFAULT_EOG):
    raw = mne.io.read_raw_edf(filepath, montage = montage, eog)
    return raw

def resample_raw_data(raw):
    decim = 3
    data =raw.get_data()
    data = np.resize(data,(64,30000))
    raw1 = mne.filter.resample(data, up=decim, npad='auto')
    ##
    ### Raw mne object is assigned and stored in a copy
    nd = Noisydata(raw1)
    ###Finding all bad channels and getting a summary
    nd.find_all_bads()
    bads = nd.get_bads(verbose=True)
    ##
    ###Bad channels are in bads so we can process the raw object
    ###We can check channel correlations
    nd._channel_correlations
    ###Checking high noise frequency per channel
    nd._channel_hf_noise
    
    
def get_average_eog(raw):
    average_eog = create_eog_epochs(raw).average()
    return average_eog

def apply_ICA(raw):
    raw.load_data()
#1 Hz high pass filtering helpful for fitting ICA 
    filtered_raw = raw.filter(1., None, n_jobs=1, fir_design='firwin')
    picks_eeg = mne.pick_types(filtered_raw.info, meg=False, eeg=True,eog=False,stim=False, exclude='bads')

##Fit ICA: Application of ICA to find the EOG artifacts in the raw data
    method='fastica'
#
##Choosing other parameters
    n_components = 62 #if float then selection by explained varience of PCA
    decim = 3 #We need sufficient statistics not all time points time saving
#
##ICA non-deterministic so we generate random state for every time the code is run and we want same decomposition with same order of components
    random_state = 60
##Defining ICA intance object
    ica_with_artifacts = ICA(n_components=n_components,method=method,random_state=random_state)
    
    reject = dict(eeg = 10e-5)
    ica = ica_with_artifacts.fit(filtered_raw,picks=picks_eeg,decim=decim,reject=reject)
    return ica