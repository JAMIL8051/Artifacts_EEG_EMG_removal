import mne
import numpy as np
import MasterModifiedKmeans as ModifiedKmeans
import MicrostateQuantifiers
import Configuration


# Function to shuffle the data: 
def shuffle_data(data, n_condition):
    for i in range(data.shape[0]):
        random_index = np.random.permutation(n_condition)
        data[i] = data[i][random_index]
    return data


# Helper function! will move to utils in future!!!
def effect(condition_0, condition_1, parameter):
    val_0 = []

    for key in condition_0[parameter]:
        val_0.append(condition_0[parameter][key])

    val_1 = []
    for key in condition_1[parameter]:
        val_1.append(condition_1[parameter][key])

    val_0, val_1 = np.asarray(val_0), np.asarray(val_1)
    # Condition_1 vs Condition_0:
    resultantEffect = val_1 - val_0
    
    return resultantEffect
# Function to find the effect using the data and microstate analysis by calculating the microstate
# map quantifiers


def findEffect(conta_data, non_conta_data, optimalNumberOfCluster, return_maps = False):

    # BDF data so a simple conversion of volt to microvolt
    conta_data = conta_data/1e-6 
    non_conta_data = non_conta_data/1e-6
    
    n_maps = optimalNumberOfCluster
    
    conta_maps, conta_labels, gfp_peaks,gev,cv = ModifiedKmeans.kmeans(conta_data, n_maps, n_runs= 10, 
                                                     maxerr=1e-6, maxiter=1000, doplot = False)
    
    
    non_conta_maps, non_conta_labels,gfp_peaks,gev,cv = ModifiedKmeans.kmeans(non_conta_data, n_maps, n_runs= 10, 
                                                             maxerr=1e-6, maxiter=1000, 
                                                             doplot = False)
    
    # Calculatin the microstate qunatifiers
    condition_0 = MicrostateQuantifiers.quantifyMicrostates(conta_maps, conta_labels)
    condition_1 = MicrostateQuantifiers.quantifyMicrostates(non_conta_maps, non_conta_labels)
    
    # The parameter variable comes from quantuify microstate function. The number of parameters can be increased or decreased
    # or modified in the MicrostateQuantifiers script.
    parameters =[]
    for key in condition_0:
        if key != 'gfp' and key != 'indices BackTraceData of Microstate classes':
            parameters.append(key)

    effectResults = {}
    for i in range(len(parameters)):
        result = effect(condition_0, condition_1, parameters[i])
        effectResults[parameters[i]+'-effect'] = result
    if return_maps:
        return effectResults, parameters, conta_maps, non_conta_maps
    return effectResults, parameters 
    
    
#Small twick to find the observed effect
def findObservedEffect(raw, rawWithArtifactsDetected, optimalNumberOfCluster):
    conta_data, times = rawWithArtifactsDetected.get_data(return_times = True)
    conta_data = conta_data

    #In case the raw object fails to work! Use the line below
    non_conta_data = raw.pick(picks = Configuration.channelList()).get_data()
    time_samples = len(times)-(len(times) % 1024)
    # +1 as the first sample beling very close to zero is ignored. This is dataset specific. User can change anytime
    non_conta_data = non_conta_data[:,1:time_samples+1]
    conta_data = conta_data[:,1:time_samples+1] 
    observedEffectResults, parameters, conta_maps, non_conta_maps = findEffect(conta_data.T, non_conta_data.T, 
                                                                               optimalNumberOfCluster, 
                                                                               return_maps = True)
    return observedEffectResults, parameters, conta_maps, non_conta_maps 

  
# We randomly assign a condition on each epochs for single subject analysis: 
# Future target on subject basis data: Randomly assign condition on to subject data: 
# See:
"""A Tutorial on Data-Driven Methods for Statistically Assessing ERP Topographies by Thomas Koenig,
    Maria Stein, Matthias Grieder, Mara Kottlow for more details.
"""

# Final implementation of randomization statistics to get the map labels

def findRandEffect(data, n_channels, n_condition, n_times, no_epochs, optimalNumberOfCluster):
    runs = Configuration.numberOfRandomRuns()
    #n_parameters = Configuration.numberOfMicrostateQuantifiers()
    # Here 3 is for the number of parameters quantifying the microstate maps. We used 3 parameters. So 3 is given.
    randEffectSize = np.zeros((runs, 3, optimalNumberOfCluster),dtype = 'float')
    
    
    while n_times<Configuration.numberOfRandomRuns():
        print('randomization run no: ', n_times)
        rand_data = shuffle_data(data, n_condition)
        rand_data = rand_data.reshape((n_condition, no_epochs, n_channels, 1024))
    
        rand_raw_non_conta_data = rand_data[0].reshape((no_epochs*1024, n_channels))
        rand_raw_conta_data = rand_data[1].reshape((no_epochs*1024, n_channels))
    
        # FUNCT1: Akta function hobe ja maps label generate kore quanitifier call korbe then 
        # duita condition er jonno quantifier diff ber kore return korbe hash map or list e
        randEffect, parameters = findEffect(rand_raw_conta_data, rand_raw_non_conta_data, optimalNumberOfCluster)
        randEffectSize[n_times, 0] = randEffect[parameters[0]+'-effect']
        randEffectSize[n_times, 1] = randEffect[parameters[1]+'-effect']
        randEffectSize[n_times, 2] = randEffect[parameters[2]+'-effect']
        
        n_times += 1

    return randEffectSize, parameters


# Finally finds the label by calculating the probability using no. of random shffles chosen
def findLabel(observedEffect, optimalNumberOfCluster, randEffectSize):
    
    microstateClass = np.zeros((optimalNumberOfCluster), dtype = 'int')
    
    for j in range(optimalNumberOfCluster):
        count = 0
        for i in range(randEffectSize.shape[0]):
            if randEffectSize[i][j] >= observedEffect[j]:
                count += 1
    
        microstateClass[j] = count
    
    microstateClassProbability = microstateClass/randEffectSize.shape[0]
    
    print('The probabilities of the 10 microstate classes:\n', microstateClassProbability)
    
    sigDiffMapLabel = []
    sigNotDiffMapLabel = []

    for i in range(len(microstateClassProbability)):
        if microstateClassProbability[i] <= 0.05:
            sigDiffMapLabel.append(i)
        else:
            sigNotDiffMapLabel.append(i)

    return sigDiffMapLabel, sigNotDiffMapLabel 


def randomizationStat(raw, rawWithArtifactsDetected, artifactualData, optimalNumberOfCluster):
    #FUNCT1:Call hobe observed effect ber korar jonno: 
    #2 times: 1 for opt_cluster1 1 for optimalNumberOfCluster 
    #chinta korbo: maps plus label duitai return koriye dictionary te store korbe:
    #ei gula hobe dictionary: sigDiffMapLabel, sigNotDiffMapLabel
    print('First run for finding the Observed Effect:')
    observedEffectResults, parameters, conta_maps, non_conta_maps = findObservedEffect(raw, 
                                                                                       rawWithArtifactsDetected, 
                                                                                       optimalNumberOfCluster)
    
    raw_conta_data, times = rawWithArtifactsDetected.get_data(return_times = True)

    channels = Configuration.channelList()
    n_channels = len(channels)

    no_epochs = len(times)//1024

    # Time samples to be taken to divide the data into epochs of 2 seconds
    time_samples = len(times) - (len(times) % 1024)

    raw_non_conta_data = raw.pick(picks = channels).get_data()
    # time_samples+1 because the first sample being very close to zero is ignored. This is dataset specific. 
    # User can change anytime. Raw non-conta data itself is a single condition so we give #1 in reshape
    raw_non_conta_data = raw_non_conta_data[:,1:time_samples+1].reshape((no_epochs, 1, n_channels, 1024))

    # Like above the same is applied for raw contaminated data.
    raw_conta_data = raw_conta_data[:,1:time_samples+1].reshape((no_epochs, 1, n_channels, 1024))

    #condition = ['non_conta','conta']
    condition = Configuration.conditionForStatAnalysis()
    n_condition = len(condition)
    data = np.concatenate((raw_non_conta_data, raw_conta_data), axis = 1)
    n_times = 0
    
    randEffectSize, parameters = findRandEffect(data, n_channels, n_condition, n_times, no_epochs, optimalNumberOfCluster)

    # Function 2: While loop theke ber hobo then ei kaj korbo. Null hypothesis er porbability 
    # calculate korbe for each micrstate class and will return: Significantly different map+label and 
    # Significantly not different map+label
    observedCountOfTimePointsEffect = observedEffectResults[parameters[0]+'-effect']
    observedOnsetOfMicrostateEffect = observedEffectResults[parameters[1]+'-effect']
    observedOffsetOfMicrostateEffect = observedEffectResults[parameters[2]+'-effect']

    countOfTimePointsRandEffectSize = randEffectSize[:,0,:]
    onsetOfMicrostateRandEffectSize = randEffectSize[:,1,:]
    offsetOfMicrostateRandEffectSize = randEffectSize[:,2,:]

    print('For the parameter: ', parameters[0])
    sigDiffMapLabel_0, sigNotDiffMapLabel_0 = findLabel(observedCountOfTimePointsEffect, optimalNumberOfCluster, countOfTimePointsRandEffectSize)
    print('For the parameter: ', parameters[1])
    sigDiffMapLabel_1, sigNotDiffMapLabel_1 = findLabel(observedOnsetOfMicrostateEffect, optimalNumberOfCluster, onsetOfMicrostateRandEffectSize)
    print('For the parameter: ', parameters[2])
    sigDiffMapLabel_2, sigNotDiffMapLabel_2 = findLabel(observedOffsetOfMicrostateEffect, optimalNumberOfCluster, offsetOfMicrostateRandEffectSize)

    labels={}
    labels[parameters[0]+'Significant'] = sigDiffMapLabel_0  
    labels[parameters[0]+'notSignificant'] = sigNotDiffMapLabel_0  
    labels[parameters[1]+'Significant'] = sigDiffMapLabel_1
    labels[parameters[1]+'notSignificant'] = sigNotDiffMapLabel_1
    labels[parameters[2]+'Significant'] = sigDiffMapLabel_2
    labels[parameters[2]+'notSignificant'] = sigNotDiffMapLabel_2
    

    return raw_non_conta_data, labels, parameters, conta_maps, non_conta_maps 



