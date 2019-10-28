import mne
import numpy as np
from pyprep.noisy import Noisydata
from mne.preprocessing import create_eog_epochs, compute_proj_eog
from mne.preprocessing import ICA
BIOSEMI_MACHINE_NAME = 'biosemi'
DEFAULT_MONTAGE_MACHINE = BIOSEMI_MACHINE_NAME + '64'
DEFAULT_EOG = ['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']


def load_montage(channel_names,
                 channel_file_path,
                 kind=DEFAULT_MONTAGE_MACHINE,
                 unit='m',
                 transform=False):
    montage = mne.channels.read_montage(kind, channel_names, channel_file_path, 
                                        unit, transform)
    return montage


def load_biosemi_montage(biosemi_channel_count=64):
    montage = mne.channels.read_montage('biosemi64')
    return montage


def load_raw_data(filepath, montage, eog=DEFAULT_EOG):
    raw = mne.io.read_raw_edf(filepath, montage, eog)
    return raw

#Finding EOG events, creating epochs, averaging
def get_average_eog(raw):
    average_eog = create_eog_epochs(raw).average()
    return average_eog

#Finding EOG artifacts: Locating peaks of EOG to spot blinks and general EOG artifacts
def find_eog_artifacts(raw, event_id):
    eog_events = mne.preprocessing.find_eog_events(raw, event_id)

# read epochs
    picks = mne.pick_types(raw.info, eeg=False, eog=True,
                           exclude='bads')
    tmin, tmax = -0.2, 4.0
    epochs = mne.Epochs(raw, eog_events, event_id, tmin, tmax, picks=picks)
    data = epochs.get_data()
    print("No. of detected EOG artifacts : %d" % len(data))
    return epochs, data


#Resampling raw data using pyprep package
def resample_raw_data(raw):
    decim = 3
    data = raw.get_data()
    data = np.resize(data, (64, 30000))
    raw1 = mne.filter.resample(data, up=decim, npad='auto')
    ##
    # Raw mne object is assigned and stored in a copy
    nd = Noisydata(raw1)
    # Finding all bad channels and getting a summary
    nd.find_all_bads()
    bads = nd.get_bads(verbose=True)

# Bad channels are in bads so we can process the raw object
# We can check channel correlations
    nd._channel_correlations
# Checking high noise frequency per channel
    nd._channel_hf_noise


# Compute SSP projections for EOG:
def compute_SSP_projection_eog(raw):
    projs, events = compute_proj_eog(raw, n_eeg=64, average=True)
    eog_projs = projs[-3:]
    return projs, eog_projs


def apply_SSP_projections(raw, eog_projs):
    raw.info['projs'] += eog_projs


def demonstrate_SSP_cleaning(raw):
    events = mne.find_events(raw, stim_channel=raw.ch_names[-1])
    reject = dict(eeg=10e-6, eog=150e6)
    # This is highly data dependent
    event_id = 10
    epochs_no_proj = mne.Epochs(raw, events, event_id, tmin=-0.2,
                                tmax=0.5, proj=False, reject=reject)
    epochs_proj = mne.Epochs(raw, events, event_id, tmin=-0.2,
                             tmax=0.5, proj=True, reject=reject)
    evoked = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.5,
                        proj='delayed', reject=reject).average()
    return epochs_no_proj, epochs_proj, evoked, events



#Application of ICA:
#Advanced part
def create_average_eog_epochs(raw):
    reject = dict(eeg=10e-6)
    picks_eeg = mne.pick_types(raw.info, meg=False, eeg=True, eog=False,
                               stim=False, exclude='bads')

# Advanced and efficient way for artifact detection
    eog_average = create_eog_epochs(
        raw, reject=dict(eeg=10e-6), picks=picks_eeg).average()
    eog_epochs = create_eog_epochs(raw, reject=reject)  # get single EOG trials

    return eog_epochs, eog_average




def apply_ICA(raw):
    raw.load_data()
# 1 Hz high pass filtering helpful for fitting ICA
    raw.filter(1., None, n_jobs=1, fir_design='firwin')
    picks_eeg = mne.pick_types(
        raw.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')

# Fit ICA: Application of ICA to find the EOG artifacts in the raw data
    method = 'fastica'

# Choosing other parameters
    n_components = 62  # if float then selection by explained varience of PCA
    decim = 3  # We need sufficient statistics not all time points time saving

# ICA non-deterministic so we generate random state for every time the code is run and we want same decomposition with same order of components
    random_state = 60

# Defining ICA intance object
    ica_with_artifacts = ICA(n_components=n_components,
                             method=method, random_state=random_state)
    reject = dict(eeg=10e-5)
    ica = ica_with_artifacts.fit(
        raw, picks=picks_eeg, decim=decim, reject=reject)

# Get single EOG trials
    eog_epochs, eog_average = create_average_eog_epochs(raw)

# Find via correlation
    eog_inds, scores = ica.find_bads_eog(eog_epochs)
    raw_copy = raw.copy().crop(0, 10)
    ica.apply(raw_copy)

    return ica, eog_inds, scores, eog_average, eog_epochs, raw_copy



