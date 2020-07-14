import mne
import numpy as np
import BackFit
import Configuration


"""
This is the interpolation script. This script starts at first with backfit operation.
Input parameters: raw = raw object having the individual subject data see MNE documentation for more details
labels = a python dictionary containing the Significantly Different and not-Different micrsotate map labels with 
the microstate quantifers as keys or ids.
conta_maps = microstate maps generated from the randomization statistical analysis, it is a numpy array having
dimension of 10 by 16. 10 represents the optimal-number of clusters/microstate class and 16 is the numebr of 
EEG-channels. 
Step 1: It backfits the micrsotate maps-data generated
from the individual-subject data, under the condition "EMG Contaminated" to the preporcessed raw data of the 
individual subject data. In this step each time point in the preporcessed raw data is labelled as a map label 
using the spatial correlation technique. So we get all time points: labelled from 0 to 9 as our optimal-number of
clusters is 10.

Step 2: The Significantly not different map labels are used to construct a subset of the preporcessed raw data.
This is done by choosing "data" from the time points that are associated with the Significantly not different map 
labels. Thus we get a numpy array having a dimension of 16 by * . Here * represents the number of times-points
that are associated with the Significantly not different map labels.

Step 3: Pretty much the concept of creating the MNE raw data object using the mne.io.RawArray function.
Step 4: Interpolating the bad channels in the raw-object as all channels are considered bad here and then
extract the interpolated data.

Step 5: In this step we concatenate the data-matrix obtained from the backfit function using the same technique 
mentioned in the previous step by using the "Significantly Different map labels" and repeat step 3 to create
the EMG artifacts removed raw data object for further validation and comparison.

"""
def excludeZeros(data):
    data = data[data!=0]
    n_times_points = len(data)-len(data)% data.shape[0]
    time_points_channel = len(data)//data.shape[0]
    excludeZeroData = data[0:n_times_points].reshape(data.shape[0],time_points_channel)

    return excludeZeroData



def interpolate(interpolateData, conta_maps, interpolateLabel):
    
    interpolateData = excludeZeros(interpolateData) 
    
    instantaneousFittedLabels, peakGfpFittedLabels = BackFit.backFit(interpolateData.T, conta_maps)
    
    # Extraction of data fitted with conta_maps using the NotSigDiffMapLabels
    
    artifactualData = np.zeros((interpolateData.shape[0], interpolateData.shape[1]),dtype='float')
    errorData = np.zeros((interpolateData.shape[0], interpolateData.shape[1]),dtype='float')
 

    for i in range(interpolateData.shape[1]):
        for j in range(len(interpolateLabel)):
            if (instantaneousFittedLabels[i] == interpolateLabel[j]) or (peakGfpFittedLabels[i] == interpolateLabel[j]):
                n_times = i
                artifactualData[:,i] = data[:,n_times]
            else:
                errorData[:,i] = data[:,n_times]
     
    artifactualData = excludeZeros(artifactualData) 
    errorData = excludeZeros(errorData)


    



    info = mne.create_info(ch_names = Configuration.channelList(), sfreq = raw.info['sfreq'], 
						ch_types= ['eeg']*data.shape[0], montage= Configuration.channelLayout())

    # Adding some more information
    info['description'] = Configuration.descibeInterpolate() 
    info['bads'] = Configuration.channelList() # Taking all the channels
    
    
    interpolateRaw = mne.io.RawArray(interpolateData, info)
    interpolateRaw = interpolateRaw.interpolate_bads(reset_bads = False, mode='accurate',origin = (0.0,0.0,0.04))

    return interpolateRaw, errorData



