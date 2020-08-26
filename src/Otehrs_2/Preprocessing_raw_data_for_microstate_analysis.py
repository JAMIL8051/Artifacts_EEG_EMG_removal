import tkinter as tk
from tkinter import filedialog
#import os
#from pathlib import Path
import mne
from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs
import numpy as np
from pyprep.noisy import Noisydata

print(__doc__)

#Tkinter package for user selection of files: DATA and Channel Location file
root = tk.Tk()
root.withdraw()

print("Please select the data file")
data_file_path  = filedialog.askopenfilename()
input_fname = data_file_path
#print("Please select the channel location file")
#channel_file_path = filedialog.askopenfilename()

# "kind" and "path" variables for mne.channels_read_montage function 
#p = Path(channel_file_path)
#k = p.parts[-1]
#d = k.find('.')
#kind = k[0:d]
#
#f = p.parts
#f=f[0:len(f)-1]
#path = os.path.join(*f) 


#Channel location file:Here we used Biosmei 64 channels machine
montage = mne.channels.read_montage('biosemi64')
#montage = mne.channels.read_montage(kind = kind, ch_names=None, path =path, unit='cm',transform=False)

#Creating the raw object using MNE package for EEG data preprocessing and we have 8 EOG channels
raw = mne.io.read_raw_edf(input_fname, montage=montage, eog = ['EXG1','EXG2','EXG3','EXG4','EXG5','EXG6','EXG7','EXG8'])

#Plotting the power spectral density of the raw data
raw.plot_psd(tmax=np.inf, fmax=500)#Frequency domain analysis

#Resampling(down sample) the data
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

###Bad channels are in bads so we can process the raw object
###We can check channel correlations
nd._channel_correlations
###Checking high noise frequency per channel
nd._channel_hf_noise
#
#
#
##EOG Artifacts identification stage:
##Finding, averaging, plotting the EOG events with the creation of the epochs
average_eog = create_eog_epochs(raw).average()
print('We found %i EOG events' % average_eog.nave)
joint_kwargs =dict(ts_args=dict(time_unit='s'))
average_eog.plot_joint(**joint_kwargs)
#
#
##Now we apply ICA on the raw data
##For filtering we load the data
raw.load_data()
#
##1 Hz high pass filtering helpful for fitting ICA 
raw.filter(1., None, n_jobs=1, fir_design='firwin')
picks_eeg = mne.pick_types(raw.info, meg=False, eeg=True,eog=False,stim=False, exclude='bads')
#
##Fit ICA: Application of ICA to find the EOG artifacts in the raw data
method='fastica'
#
##Choosing other parameters
n_components = 62 #if float then selection by explained varience of PCA
decim = 3 #We need sufficient statistics not all time points time saving
#
##ICA non-deterministic so we generate random state for every time the code is run and we want same decomposition with same order of components
random_state = 60
#
##Defining ICA intance object
ica = ICA(n_components=n_components,method=method,random_state=random_state)
print(ica)
#
##Avoiding of fitting ICA on crazy artifacts
reject = dict(eeg = 10e-5)
ica.fit(raw,picks=picks_eeg,decim=decim,reject=reject)
print(ica)
#
##Plotting of the ICA components
ica.plot_components()#Some potential bad guys can be spotted
#
##First component 0 have a close look
ica.plot_properties(raw,picks=0)
#
##Data was fitlered so less informative sprectrum plot. Hence we change:
ica.plot_properties(raw,picks=0,psd_args={'fmax': 250.})

#Multiple different components at once
ica.plot_properties(raw,picks=[1,2,3],psd_args={'fmax':500.})

