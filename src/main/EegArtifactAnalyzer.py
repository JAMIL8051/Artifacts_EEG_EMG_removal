import ConfigFile
import EegPreprocessor as preprocessor
import PowerAnalysis
import numpy as np
import mne

#Function to detect the EMG artifacts from the frontalis and temporal brain region after preprocessing
#using the power analysis in the 45-70Hz frequency band

def detectArtifacts(filepath):
	raw = preprocessor.preprocessRawData(filepath)
	dataWithArtifactsDetected, ch_names_combined, artifactualData = PowerAnalysis.identifyArtifacts(raw)
	
	return raw, dataWithArtifactsDetected, ch_names_combined, artifactualData


#Function to remove the EMG artifacts using microstate analysis and
#randomization statistics

def formatForMicrostate(raw, dataWithArtifactsDetected, ch_names_combined):
	dataWithArtifactsDetected = dataWithArtifactsDetected.reshape((len(ch_names_combined), 100, 1024))
	
	channels, epochs, times = np.nonzero(dataWithArtifactsDetected != 0)
	
	dataWithArtifactsDetected = dataWithArtifactsDetected[dataWithArtifactsDetected!= 0]
	
	n_channels = len(ch_names_combined)
	n_time_points = len(dataWithArtifactsDetected)-(len(dataWithArtifactsDetected) % n_channels)
	time_points_channel = len(times)//n_channels
	
	dataWithArtifactsDetected = dataWithArtifactsDetected[0:n_time_points].reshape(time_points_channel, 
																				n_channels)
	
	sampling_rate = raw.info['sfreq']
	info = mne.create_info(ch_names = ch_names_combined, sfreq = sampling_rate, 
						ch_types= ['eeg']*n_channels)
	
	dataWithArtifactsDetectedRaw = mne.io.RawArray(dataWithArtifactsDetected.T, info)
	
	return dataWithArtifactsDetectedRaw


def removeArtifacts(dataWithArtifactsDetectedRaw, artifactualData, backfit=True, 
					interpolate = False):
	# Doing the microstate analysis to generate the microstate classes
	microstateResults = MicrostateAnalyzer.analyzeMicrostate(dataWithArtifactsDetectedRaw, 
														  artifactualData)

	#
	randStatResults = RandomizationStatistics.randomizationStat(microstate_results)

	#
	if backfit:
		backFitResult = BackFit.backFit(rand_stat_results)

	#
	if interpolate:
		interploateResult = interpolate(rand_stat_results)
	
	return backFitResult, interploateResult


#Function to detect EMG contaminated EEG segments after standard preprocessing of the data and then conducting
#power analysis in the 45-70Hz frequency band and to remove those artifactual segments by doing micrsotate 
#analysis and randomization statistics.

def detectAndRemoveEegArtifact(filepath, backfit=True, interpolate= False, validation=False, 
								   comparison= False, visualize=False):

	raw, dataWithArtifactsDetected, ch_names_combined, artifactualData = detectArtifacts(filepath)
	     
	dataWithArtifactsDetectedRaw = formatForMicrostate(raw, dataWithArtifactsDetected, ch_names_combined)

	removalResult = removeArtifact(raw, artifactualData, backfit, interpolate)
	
	if validation:
		validateWithSimulatedData()

	if comparison:
		compareWithIcaMara()

	if visualize:
		visualization()


	return None	