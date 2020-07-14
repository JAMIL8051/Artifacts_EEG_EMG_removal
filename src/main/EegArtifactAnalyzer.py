import EegPreprocessor as preprocessor
import PowerAnalysis
import Configuration
import MicrostateAnalyzer
import ModifiedKmeans
import RandomizationStatistics
import BackFit
import Interpolate
import numpy as np
import mne
from mne.viz import set_3d_title, set_3d_view
import matplotlib.pyplot as plt

 
# This script start from the function: "detectAndRemoveEegArtifact". This function detects the EMG artifacts due to 
# frontalis and temporalis scalp muscle using power analysis in 45-70 Hz technique. For details follow:

"""A Detection Scheme for Frontalis and Temporalis Muscle EMG Contamination of EEG Data
Michael J. Fu∗, Janis J. Daly†, M. Cenk C¸ avus¸o˘glu‡ 
∗Department of Electrical Engineering and Computer Science, 
Case Western Reserve University, Cleveland, Ohio 44106 Email: mjfu@case.edu
†Department of Neurology, Case Western Reserve University School of Medicine, 
Cleveland, Ohio 44106 Stroke Motor Control, Motor Laboratory, and FES Center of Excellence, 
LSCDVA Medical Center, Cleveland Ohio 44106 Email: jjd17@case.edu 
‡Department of Electrical Engineering and Computer Science, Case Western Reserve University, 
Cleveland, Ohio 44106Email: cavusoglu@case.edu
""" 

#Function to format the data for microstate analysis
def formatForMicrostate(raw, dataWithArtifactsDetected, ch_names_combined):
	dataWithArtifactsDetected = dataWithArtifactsDetected.reshape((len(ch_names_combined), 100, 1024))
	
	channels, epochs, times = np.nonzero(dataWithArtifactsDetected != 0)
	
	dataWithArtifactsDetected = dataWithArtifactsDetected[dataWithArtifactsDetected!= 0]
	
	n_channels = len(ch_names_combined)
	n_time_points = len(dataWithArtifactsDetected)-(len(dataWithArtifactsDetected) % n_channels)
	time_points_channel = len(times)//n_channels
	
	dataWithArtifactsDetected = dataWithArtifactsDetected[0:n_time_points].reshape(time_points_channel, 
																				n_channels)
	
	info = mne.create_info(ch_names = ch_names_combined, sfreq = raw.info['sfreq'], 
						ch_types= ['eeg']*len(ch_names_combined))
	
	dataWithArtifactsDetectedRaw = mne.io.RawArray(dataWithArtifactsDetected.T, info)

	return dataWithArtifactsDetectedRaw


#Function to detect the EMG artifacts from the frontalis and temporal brain region after preprocessing
#using the power analysis in the 45-70Hz frequency band
def detectArtifacts(filepath):
	raw = preprocessor.preprocessRawData(filepath)
	artifactualData, finalEmgData1, ch_names_combined = PowerAnalysis.identifyArtifacts(raw)
	
	return raw, ch_names_combined, artifactualData, finalEmgData1


# Please start from the bottom to top reading approach. As in python sub-functions inside a big function needs to be on the top
# of the big function
# Function for plotting the optimal number of microstate maps obtained after microstate analysis        
# Function to remove EMG artifacts using microstate analysis and randomization statistics
def separateLabels(labels):
    SigDiffMapLabel = []
    SigNotDiffMapLabel = []
    string = 'notSignificant'

    for key in labels:
        if string not in key:
            SigDiffMapLabel.append(labels[key])
        else:
            SigNotDiffMapLabel.append(labels[key])
    
    SigDiffMapLabel = set(list(set(sum(SigDiffMapLabel, []))))
    SigNotDiffMapLabel = set(list(set(sum(SigNotDiffMapLabel, []))))

    interpolateLabel = SigNotDiffMapLabel - SigDiffMapLabel
    interpolateLabel = list(interpolateLabel)


    return SigDiffMapLabel, interpolateLabel





def removeArtifacts(raw, rawWithArtifactsDetected, artifactualData, trainDataPath, backfit=True, interpolate = True):

	# First step: Find optimal number of microstate classes
	# Doing the microstate analysis to generate the optimal number of microstate classes
	# optimalMaps, optimalNumberOfCluster = MicrostateAnalyzer.analyzeMicrostate(trainDataPath)
	
	
	# Conduct randomized statistics with help of quantifiers of microstate classes or maps to generate the 
	# significantly different maps with labels for backfit and significantly not different ones for interpolation
	raw_non_conta_data, labels, parameters, conta_maps, non_conta_maps = RandomizationStatistics.randomizationStat(raw, 
																					rawWithArtifactsDetected, 
																					artifactualData, 
																					optimalNumberOfCluster = 10)

	# Separting the labels parameter into two other ones: sigDiffmaplabel and interpolateLabel for the 
	# backfit and interpolation part.
	SigDiffMapLabel, interpolateLabel = separateLabels(labels)




	
	# Here the backfit process is done in two ways. First, the whole individual raw single-subject data is taken
	# from the raw object. This data is 16 by 128000 ndarray. 16 channels represents the channelList in the 
	# Configuration file. Second, the conta_maps, non_conta_maps parameters are fitted to the instantaneous data 
	# and on the gfp peaks of the data. For three micrsotate-quantifiers the laebls are computed and the final 
	# results are obtained.
	  
	if backfit:
		data = raw.pick(picks = Configuration.channelList()).get_data()
		data = data[:,1:]# Removing the first time-sample of all the channels as it is very very close to zero 
		# and hence an outlier
		data = data/1e-6
		# Back fitting the optimal maps obtained from average data
		
		intantaneousEEGLabel, peakGfpLabel = BackFit.backFit(data.T, non_conta_maps)
		


		# Can be done in another way: Call Modified K means/findEffect function and take maps variable condition wise and 
		# fit the maps using the sigDiffMapLabel as obtained from randomization 
		 

    # Extraction of data for fitted with non_conta_maps using the SigDiffMapLabels and sepaarting data for 
    # interpolation
		backfitData = np.zeros((data.shape[0], data.shape[1]),dtype='float')
		interpolateData = np.zeros((data.shape[0], data.shape[1]),dtype='float')


		for i in range(data.shape[1]):
			for j in range(len(SigDiffMapLabels)):
				if (intantaneousEEGLabel[i] == SigDiffMapLabels[j]) or (peakGfpLabel[i] == SigDiffMapLabels[j]):
					n_times = i
					backfitData[:,i] = data[:,n_times]
				else:
					interpolateData[:,i] = data[:,n_times]

		backfitData = Interpolate.excludeZeros(backfitData)
		
		

	# Function for interpolation of artifactual data obtained using the contaminated maps with SigNotDiffMap labels
	if interpolate:
		interpolateRaw = Interploate.interpolate(interpolateData, conta_maps, interpolateLabel)
		
		
		# Can be done in another way: Same strategy as used in backfit of data.
	
	return backfitData, interpolateRaw

#conta_data, times = rawWithArtifactsDetected.get_data(return_times = True)
		#time_samples = len(times) - (len(times) % 1024)
		#non_conta_data = data[:,1:time_samples+1]
		#non_conta_data = non_conta_data/1e-06
		#optimalMaps, mapLabels, gfp_peaks, gev, cv = ModifiedKmeans.kmeans(non_conta_data.T, n_maps = 10, n_runs = 10, maxerr = 1e-06, maxiter = 1000, doplot=False) 

#Function to detect EMG contaminated EEG segments after standard preprocessing of the data and 
#then conducting power analysis in the 45-70Hz frequency band and to remove those artifactual 
#segments by doing micrsotate analysis and randomization statistics.

def detectAndRemoveEegArtifact(filepath, trainDataPath, backfit=True, interpolate= True, 
							   validation=False, comparison= False, visualize=False):

	#Function to detect artifacts
	raw, ch_names_combined, artifactualData, finalEmgData2 = detectArtifacts(filepath)


	# Format for EEG micrsotate analysis as well as randomization statistics     
	# rawWithArtifactsDetected = formatForMicrostate(raw, dataWithArtifactsDetected, ch_names_combined)
	finalEmgData2Raw = formatForMicrostate(raw, finalEmgData2, list(set(ch_names_combined)))
	
	
	# Function to remove the EMG artifacts
	backfitData, interpolateRaw = removeArtifacts(raw, finalEmgData2Raw, artifactualData, trainDataPath, backfit, 
											     interpolate)
	# Data recontruction
	emgFreeData = np.concatenate((backfitData,interpolateRaw.get_data()),axis = 0)
	# End of data analysis and the algorithm


	# For validation purpose with simulated EEG data
	if validation:
		validateWithSimulatedData()

	# For standard comparison with other method like: ICA+MARA 
	if comparison:
		compareWithIcaMara()

	# Display of all the results and generation of report of the analysis
	if visualize:
		montage = mne.channels.make_standard_montage(Configuration.channelLayout())
		info = mne.create_info(ch_names=montage.ch_names, sfreq= raw.info['sfreq'], ch_types='eeg', montage=montage)
		sphere = mne.make_sphere_model(r0='auto', head_radius='auto', info=info)
		fig = mne.viz.plot_alignment(show_axes=True, dig='fiducials', surfaces='head', bem=sphere, info=info)# Plot options
		set_3d_view(figure=fig, azimuth=135, elevation=80)
		set_3d_title(figure=fig, title=current_montage)
		raw.plot()

	return None	