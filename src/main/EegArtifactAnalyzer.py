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
	
	sampling_rate = raw.info['sfreq']
	info = mne.create_info(ch_names = ch_names_combined, sfreq = sampling_rate, 
						ch_types= ['eeg'], montage = 'biosemi64')
	
	dataWithArtifactsDetectedRaw = mne.io.RawArray(dataWithArtifactsDetected.T, info)
	
	return dataWithArtifactsDetectedRaw


def removeArtifacts(raw, rawWithArtifactsDetected, artifactualData, backfit=True, 
					interpolate = False):
	# First step: Find optimal number of microstate classes
	# Doing the microstate analysis to generate the optimal number of microstate classes
	opt_cluster1, optimalNumberOfCluster = MicrostateAnalyzer.analyzeMicrostate(raw)

	# Conduct randomized statistics with help of quantifiers of microstate classes or maps 
	# to generate the significantly different maps with labels for backfit and 
	# significantly not different ones for interpolation
	sigDiffMapLabel, sigNotDiffMapLabel = RandomizationStatistics.randomizationStat(raw, 
											rawWithArtifactsDetected, 
											artifactualData, 
											opt_cluster1, 
											optimalNumberOfCluster)

	# Main criteria to preserve the data epochs or microstates
	if backfit:
		backFitResult = BackFit.backFit(raw.get_data(), sigDiffMapLabel)

	# Optional for now!!
	if interpolate:
		interploateResult = interpolate(sigNotDiffMapLabel)
	
	return backFitResult, interploateResult


#Function to detect EMG contaminated EEG segments after standard preprocessing of the data and then conducting
#power analysis in the 45-70Hz frequency band and to remove those artifactual segments by doing micrsotate 
#analysis and randomization statistics.

def detectAndRemoveEegArtifact(filepath, backfit=True, interpolate= False, validation=False, 
								   comparison= False, visualize=False):

	#Function to detect artifacts
	raw, dataWithArtifactsDetected, ch_names_combined, artifactualData = detectArtifacts(filepath)

	# Format for EEG micrsotate analysis as well as randomization statistics     
	rawWithArtifactsDetected = formatForMicrostate(raw, dataWithArtifactsDetected, ch_names_combined)
	
	# Function to remove the EMG artifacts
	backFitResult, interploateResult = removeArtifact(raw, rawWithArtifactsDetected, 
												   artifactualData, backfit, interpolate)
	
	if validation:
		validateWithSimulatedData()

	if comparison:
		compareWithIcaMara()

	if visualize:
		visualization()


	return None	