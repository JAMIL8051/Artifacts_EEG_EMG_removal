import EegPreprocessor as preprocessor
import PowerAnalysis


#Function to detect the EMG artifacts from the frontalis and temporal brain region after preprocessing
#using the power analysis in the 45-70Hz frequency band

def detectArtifacts(filepath):
	raw = preprocessor.preprocessRawData(filepath)
	#dataWithArtifactsDetected, dataWithArtifactsDetectedRaw = PowerAnalysis.identifyArtifacts(raw)
	finalEmgData = PowerAnalysis.identifyArtifacts(raw)
	#return dataWithArtifactsDetected, dataWithArtifactsDetectedRaw
	return finalEmgData

#Function to remove the EMG artifacts using microstate analysis and
#randomization statistics

def removeArtifacts(dataWithArtifactsDetected, backfit=True, interpolate = False):
	# Doing the microstate analysis to generate the microstate classes
	microstateResults = MicrostateAnalysis.microstateAnalysis(dataWithArtifactsDetected)

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
	detectionResults = detectArtifacts(filepath)

	removalResult = removeArtifact(detection_results, backfit,
								   interpolate)
	
	if validation:
		validateWithSimulatedData()

	if comparison:
		compareWithIcaMara()

	if visualize:
		visualization()


	return None	