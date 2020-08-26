import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.baseline import rescale
from mne.stats import _bootstrap_ci
from mne.decoding import UnsupervisedSpatialFilter

import tkinter as tk
from tkinter import filedialog

from sklearn.decomposition import PCA, FastICA

print(__doc__)
#The first part computes PCA and ICA of evoked or epochs found in the given data
#Generating user interface window
root = tk.Tk()
root.withdraw()

print("Please select the data file")
path_data = filedialog.askopenfilename()
print("Please select the channel location file")
path_loc = filedialog.askopenfilename()


def read_data(path):
    with open(path, 'r') as f:
       lines = f.readlines()
    matrix = []
    for line in lines:
        res = []
        temp = line.split(' ')
        for num in temp:
            if num:
                res.append(float(num))
        matrix.append(res)
    return np.asarray(matrix)

def read_pos_data(path=None):
    if path is None:
        path = 'Cap63.locs'
    with open(path, 'r') as f:
        lines = f.readlines()
    res = []
    for line in lines:
        temp = line.split('\t')
        row = []
        if len(temp) > 1:
            row.append(float(temp[1].strip()))
            row.append(float(temp[2].strip()))
            row.append(temp[3].strip())
        if row:
            res.append(row)
    return np.asarray(res)


data = read_data(path_data)
loc = read_pos_data(path_loc)
n_channels = 63
sampling_rate = 500
info = mne.create_info(ch_names=loc[:, 2].tolist(), sfreq=sampling_rate, ch_types=['eeg']*63)
print(info)

revdata = data.T
print(revdata.shape)

raw = mne.io.RawArray(revdata, info)
tmin, tmax = -0.1, 0.3
event_id = dict(aud_l = 1,aud_r=2, vis_l=3, vis_r=4)
events = mne.find_events(raw,stim_channel='POz',shortest_event=1)
picks = mne.pick_types(raw.info,meg=False,eeg=True,stim=False,eog=False)
epochs = mne.Epochs(raw,events,event_id,tmin,tmax,proj=False,
                    picks=picks,baseline= None,preload=True,
                    verbose=False)
X=epochs.get_data()

#PCA application for  transformation of Data
pca = UnsupervisedSpatialFilter(PCA(63), average=False)
pca_data =pca.fit_transform(X)
ev = mne.EvokedArray(np.mean(pca_data,axis=0),mne.create_info(63,epochs.info['sfreq'],
                     ch_types=['eeg']*63),tmin=tmin)
ev.plot(show=False, window_title="PCA", time_unit ='s')

#ICA computation with no averaging
ica = UnsupervisedSpatialFilter(FastICA(63),average=False)
ica_data = ica.fit_transform(X)

ev1 = mne.EvokedArray(np.mean(ica_data,axis=0),
                      mne.create_info(63,epochs.info['sfreq'],
                                      ch_types=['eeg']*63),tmin=tmin)
ev1.plot(show=False,window_title='ICA',time_unit='s')
plt.show()


#The second part shows how to explore spectrally localized effect in the data
iter_freqs=[
        ('Theta',4,7),
        ('Alpha',8,12),
        ('Beta',13,25),
        ('Gamma',30,45)]
#Setting epoching parameters
event_id1, tmin1, tmax1 = 1, -1., 3.
baseline = None

frequency_map =list()

for band, fmin, fmax in iter_freqs:
    raw = mne.io.RawArray(revdata,info)
    raw.pick_types(meg='grad', eeg=True, eog=True)  # we just look at gradiometers
    picks = mne.pick_types(raw.info, eeg=True,stim=False,eog=True)
    # bandpass filter and compute Hilbert
    raw.filter(fmin, fmax, n_jobs=1,  # use more jobs to speed up.
               l_trans_bandwidth=1,  # make sure filter params are the same
               h_trans_bandwidth=1,  # in each band and skip "auto" option.
               fir_design='firwin')
    raw.apply_hilbert(picks = picks,envelope=False, n_jobs=1, n_fft=63000,verbose=None )

    epochs = mne.Epochs(raw, events, event_id=event_id1, tmin=tmin1, tmax=tmax1,
                        baseline=baseline,preload=True,verbose=False)
    # remove evoked response and get analytic signal (envelope)
    epochs.subtract_evoked()  # for this we need to construct new epochs.
    epochs = mne.EpochsArray(
        data=np.abs(epochs.get_data()), info=epochs.info, tmin=epochs.tmin)
    # now average and move on
    frequency_map.append(((band, fmin, fmax), epochs.average()))

    
#Computation of Global field power    
fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True, sharey=True)
colors = plt.get_cmap('winter_r')(np.linspace(0, 1, 4))
for ((freq_name, fmin, fmax), average), color, ax in zip(
        frequency_map, colors, axes.ravel()[::-1]):
    times = average.times * 1e3
    gfp = np.sum(average.data ** 2, axis=0)
    gfp = mne.baseline.rescale(gfp, times, baseline=(None, 0))
    ax.plot(times, gfp, label=freq_name, color=color, linewidth=2.5)
    ax.axhline(0, linestyle='--', color='grey', linewidth=2)
    ci_low, ci_up = _bootstrap_ci(average.data, random_state=0,
                                  stat_fun=lambda x: np.sum(x ** 2, axis=0))
    ci_low = rescale(ci_low, average.times, baseline=(None, 0))
    ci_up = rescale(ci_up, average.times, baseline=(None, 0))
    ax.fill_between(times, gfp + ci_up, gfp - ci_low, color=color, alpha=0.3)
    ax.grid(True)
    ax.set_ylabel('GFP')
    ax.annotate('%s (%d-%dHz)' % (freq_name, fmin, fmax),
                xy=(0.95, 0.8),
                horizontalalignment='right',
                xycoords='axes fraction')
    ax.set_xlim(-1000, 3000)

axes.ravel()[-1].set_xlabel('Time [ms]')
    
    