import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA, FastICA

print(__doc__)

#Multiple artifact rejection algorithm
def mara(raw):
    raw.filter(1,40, fir_design = 'firwin')
    event_id = {'2-back_spatial': 20}

#    event_id = {'1-back_spatial': 10, '2-back_spatial': 20, '3-back_spatial': 30, '1-back_nonspatial': 40,
#            '2-back_nonspatial': 50, '3-back_nonspatial': 60}
#    
    events = mne.find_events(raw, stim_channel = raw.ch_names[-1])
    picks_eeg = mne.pick_types(raw.info, eeg = True, stim =False, eog = False)
    epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=4.5, proj =False, picks=picks_eeg, baseline=None,preload=True,verbose=False)
    X = epochs.get_data()
    #Using PCA to transform data on the average i.e. evoked response
    pca = UnsupervisedSpatialFilter(PCA(64), average = False)
    pca_transformed_data = pca.fit_transform(X)
    
    #just for plotting use these two lines
    tmin = -0.1
    ev = mne.EvokedArray(np.mean(pca_transformed_data, axis = 0), mne.create_info(64, epochs.info['sfreq'], ch_types='eeg'), tmin=tmin)
    ev.plot(show=False, window_title ="PCA", time_unit ='s')
 
#We apply TDSEP algorithm reference from Joyce 2004
    
    
    
    
    
#We apply ICA on the pca_data so creating pca_raw object    
#    ica = UnsupervisedSpatialFilter(FastICA(30, max_iter = 500), average=False)
#    ica_data = ica.fit_transform(pca_data1)
#    ev1 = mne.EvokedArray(np.mean(ica_data, axis=0),
#                      mne.create_info(30, epochs.info['sfreq'],
#                                      ch_types='eeg'), tmin=tmin)
#    ev1.plot(show=True, window_title='ICA', time_unit='s')
#
#    plt.show()
    return pca_transformed_data
    
    
    
    
    