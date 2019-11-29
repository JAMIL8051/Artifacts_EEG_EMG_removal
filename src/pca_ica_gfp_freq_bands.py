import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.baseline import rescale
from mne.stats import bootstrap_confidence_interval
from mne.decoding import UnsupervisedSpatialFilter
from scikit-learn import sklearn.decomposition import PCA, FastICA


#Function to implement PCA, ICA and frequncy spectral analysis using Global field power
def pca_ica_freq_bands_gfp(raw):
    tmin, tmax = 0, 4.5
    event_id = 20
#    event_id = {'1-back_spatial': 10, '2-back_spatial': 20, '3-back_spatial': 30, '1-back_nonspatial': 40,
#            '2-back_nonspatial': 50, '3-back_nonspatial': 60}
    
    events = mne.find_events(raw,stim_channel='STI 014')
    picks = mne.pick_types(raw.info,meg=False,eeg=True,stim= True,eog=False)
    epochs = mne.Epochs(raw,events,event_id,tmin,tmax,proj=False,
                    picks=picks,baseline= None,preload=True,
                    verbose=False)
    X = epochs.get_data()
    #PCA application for  transformation of Data
    pca = UnsupervisedSpatialFilter(PCA(63), average=False)
    pca_data =pca.fit_transform(X)
    ev = mne.EvokedArray(np.mean(pca_data,axis=0),mne.create_info(63,epochs.info['sfreq'],
                     ch_types=['eeg']*63),tmin=tmin)
    ev.plot(show=False, window_title="PCA", time_unit ='s')
    
    #ICA computation with no averaging
    ica = UnsupervisedSpatialFilter(FastICA(64),average=False)
    ica_data = ica.fit_transform(X)
    ev1 = mne.EvokedArray(np.mean(ica_data,axis=0), mne.create_info(64,epochs.info['sfreq'],
                                      ch_types=['eeg']*64),tmin=tmin)
    ev1.plot(show=False,window_title='ICA',time_unit='s')
    plt.show()
    
    #The second part shows how to explore spectrally localized effect in the data
    iter_freqs=[
        ('Theta',4,7),
        ('Alpha',8,12),
        ('Beta',13,25),
        ('Gamma',30,45)]
    #Setting epoching parameters

    baseline = None

    frequency_map =list()

    for band, fmin, fmax in iter_freqs:
        raw = mne.io.RawArray(raw.get_data(),raw.info)
        raw.pick_types(eeg=True, eog=True)  # we just look at EEG AND EOG
        
        # bandpass filter and compute Hilbert
        raw.filter(fmin, fmax, n_jobs=1,  # use more jobs to speed up.
               l_trans_bandwidth=1,  # make sure filter params are the same
               h_trans_bandwidth=1,  # in each band and skip "auto" option.
               fir_design='firwin')
        
#Epochs
        epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax,
                        baseline=baseline,reject =dict(eeg =10e-6, eog = 350e-6),preload=True)
        # remove evoked response and get analytic signal (envelope)
        epochs.subtract_evoked()  # for this we need to construct new epochs.
        # get analytic signal (envelope)
        epochs.apply_hilbert(envelope=True)
        epochs = mne.EpochsArray(
        data=np.abs(epochs.get_data()), info=epochs.info, tmin=epochs.tmin)
    # now average and move on
        frequency_map.append(((band, fmin, fmax), epochs.average()))
        del epochs
    del raw

    
#Computation of Global field power
    def stat_fun(x):
        "Return sum of squares"
        return np.sum(x ** 2, axis =0)
#We now plot    
    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True, sharey=True)
    colors = plt.get_cmap('winter_r')(np.linspace(0, 1, 4))
    for ((freq_name, fmin, fmax), average), color, ax in zip(
        frequency_map, colors, axes.ravel()[::-1]):
        times = average.times * 1e3
        gfp = np.sum(average.data ** 2, axis=0)
        gfp = mne.baseline.rescale(gfp, times, baseline=(None, 0))
        ax.plot(times, gfp, label=freq_name, color=color, linewidth=2.5)
        ax.axhline(0, linestyle='--', color='grey', linewidth=2)
        ci_low, ci_up = bootstrap_confidence_interval(average.data, random_state=0,
                                  stat_fun = stat_fun)
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
    