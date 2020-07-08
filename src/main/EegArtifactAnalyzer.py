import EegPreprocessor as preprocessor
import PowerAnalysis
import Configuration
import MicrostateAnalyzer
import RandomizationStatistics
import BackFit
import numpy as np
import mne
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
# Function for plotting the optimal number of microstate maps obtained after microstate 
# analysis

        
# Function to remove EMG artifacts using microstate analysis and randomization statistics
def removeArtifacts(raw, rawWithArtifactsDetected, artifactualData, trainDataPath, backfit=True, interpolate = False):

	# First step: Find optimal number of microstate classes
	# Doing the microstate analysis to generate the optimal number of microstate classes
	# optimalMaps, optimalNumberOfCluster = MicrostateAnalyzer.analyzeMicrostate(trainDataPath)
	
	
	
	
	
	# Conduct randomized statistics with help of quantifiers of microstate classes or maps to generate the 
	# significantly different maps with labels for backfit and significantly not different ones for interpolation
	raw_non_conta_data, labels, parameters = RandomizationStatistics.randomizationStat(raw, 
																					rawWithArtifactsDetected, 
																					artifactualData, 
																					optimalNumberOfCluster = 10)

	# Main criteria to preserve the data epochs or microstates
	#individualSubjectdata = raw.pick(picks = Configuration.channelList()).get_data()

	if backfit:
		backFitResult_0 = BackFit.backFit(raw_non_conta_data, optimalMaps, labels, parameters[0])
		backFitResult_1 = BackFit.backFit(raw_non_conta_data, optimalMaps, labels, parameters[1])
		backFitResult_2 = BackFit.backFit(raw_non_conta_data, optimalMaps, labels, parameters[2])
		backFitResultFullData = BackFit.backFit(individualSubjectdata, optimalMaps, sigDiffMapLabel)
		
		# Can be done in another way: Call Modified K means/findEffect function and take maps variable condition wise and 
		# fit the maps using the sigDiffMapLabel as obtained from randomization 
		

	# Optional for now!!
	if interpolate:
		interploateResult = interpolate(raw_non_conta_data, optimalMaps, sigNotDiffMapLabel)
		interploateResultFullData = interpolate(individualSubjectdata, sigNotDiffMapLabel)
		
		# Can be done in another way: Same strategy as used in backfit of data.
	
	return backFitResult, backFitResultFullData, interploateResult, interploateResultFullData


#Function to detect EMG contaminated EEG segments after standard preprocessing of the data and 
#then conducting power analysis in the 45-70Hz frequency band and to remove those artifactual 
#segments by doing micrsotate analysis and randomization statistics.

def detectAndRemoveEegArtifact(filepath, trainDataPath, backfit=True, interpolate= False, 
							   validation=False, comparison= False, visualize=False):

	#Function to detect artifacts
	raw, ch_names_combined, artifactualData, finalEmgData2 = detectArtifacts(filepath)


	# Format for EEG micrsotate analysis as well as randomization statistics     
	# rawWithArtifactsDetected = formatForMicrostate(raw, dataWithArtifactsDetected, ch_names_combined)
	finalEmgData2Raw = formatForMicrostate(raw, finalEmgData2, list(set(ch_names_combined)))
	
	
	# Function to remove the EMG artifacts
	backFitResult, backFitResultFullData, interploateResult, interploateResultFullData = removeArtifacts(raw, 
																										finalEmgData2Raw,
																										artifactualData,
																										trainDataPath,
																										backfit, 
																										interpolate)
	# End of data analysis


	# For validation purpose with simulated EEG data
	if validation:
		validateWithSimulatedData()

	# For standard comparison with other method like: ICA+MARA 
	if comparison:
		compareWithIcaMara()

	# Display of all the results and generation of report of the analysis
	if visualize:
		visualization()

	return None	