import mne
import numpy as np
import ModifiedKmeans
import MicrostateQuantifiers
import Configuration


# Function to shuffle the data: 
def shuffle_data(data, n_condition):
    for i in range(data.shape[0]):
        random_index = np.random.permutation(n_condition)
        data[i] = data[i][random_index]
    return data


# Function to find the effect using the data and microstate analysis by calculating the microstate
# map quantifiers
def findEffect(conta_data, non_conta_data, optimalNumberOfCluster):
    n_maps = optimalNumberOfCluster
    
    conta_maps, conta_labels, gfp_peaks,gev,cv = ModifiedKmeans.kmeans(conta_data, n_maps, n_runs= 50, 
                                                     maxerr=1e-6, maxiter=1000, doplot = False)
    
    
    non_conta_maps, non_conta_labels,gfp_peaks,gev,cv = ModifiedKmeans.kmeans(non_conta_data, n_maps, n_runs= 50, 
                                                             maxerr=1e-6, maxiter=1000, 
                                                             doplot = False)
    
    # Calculatin the microstate qunatifiers
    condition_0 = MicrostateQuantifiers.quantifyMicrostates(conta_maps, conta_labels)
    condition_1 = MicrostateQuantifiers.quantifyMicrostates(non_conta_maps, non_conta_labels)
    
    
    # Choosing only one qunatifier for simplification
    val_0=[]

    for key in condition_0['Count of time points']:
        val_0.append(condition_0['countOfTimePoints'][key])

    val_1 = []
    for key in condition_1['Count of time points']:
        val_1.append(condition_1['countOfTimePoints'][key])

    val_0, val_1 = np.asarray(val_0), np.asarray(val_1) 
    effect = val_0-val_1

    
    return effect


#Small twick to find the observed effect
def findObservedEffect(raw, rawWithArtifactsDetected, optimalNumberOfCluster):
    conta_data = rawWithArtifactsDetected.get_data().T
    non_conta_data = raw.pick(picks = rawWithArtifactsDetected.ch_names).get_data().T
    
    observedEffect = findEffect(conta_data, non_conta_data, optimalNumberOfCluster)
    

    return observedEffect


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
    
    sigDiffMapLabel = []
    sigNotDiffMapLabel = []

    for i in range(len(microstateClassProbability)):
        if microstateClassProbability[i] <=0.05:
            sigDiffMapLabel.append(i)
        else:
            sigNotDiffMapLabel.append(i)
    return sigDiffMapLabel, sigNotDiffMapLabel 


# We randomly assign a condition on each epochs for single subject analysis: 
# Future target on subject basis data: Randomly assign condition on to subject data: 
# See:
"""A Tutorial on Data-Driven Methods for Statistically Assessing ERP Topographies by Thomas Koenig,
    Maria Stein, Matthias Grieder, Mara Kottlow for more details.
"""

# Final implementation of randomization statistics to get the map labels

def randomizationStat(raw, rawWithArtifactsDetected, artifactualData, optimalNumberOfCluster):
    #FUNCT1:Call hobe observed effect ber korar jonno: 
    #2 times: 1 for opt_cluster1 1 for optimalNumberOfCluster 
    #chinta korbo: maps plus label duitai return koriye dictionary te store korbe:
    #ei gula hobe dictionary: sigDiffMapLabel, sigNotDiffMapLabel
    observedEffect = findObservedEffect(raw, rawWithArtifactsDetected, optimalNumberOfCluster)
    

    raw_conta_data, times = rawWithArtifactsDetected.get_data(return_times = True)

    channels = rawWithArtifactsDetected.ch_names
    n_channels = len(channels)

    #times = 12185

    no_epochs = len(times)//1024

    # Time samples to be taken to divide the data into epochs of 2 seconds
    time_samples = times - (times % 1024)

    raw_non_conta_data = raw.pick(picks = channels).get_data()
    # Raw non-conta data itself is a single condition so we give #1
    raw_non_conta_data = raw_non_conta_data[:,1:time_samples].reshape((no_epochs, 1, n_channels, 1024))

    # Raw conta data itself is a single condition so we give #1
    raw_conta_data = raw_conta_data[:,1:time_samples].reshape((no_epochs, 1, n_channels, 1024))

    #condition = ['non_conta','conta']
    condition = Configuration.condition()
    n_condition = len(condition)
    data = np.concatenate((raw_non_conta_data, raw_conta_data), axis = 1)
    
    n_times = 0

    randEffectSize = np.zeros((1000, optimalNumberOfCluster),dtype = 'float')

    while n_times<1000:
    
        rand_data = shuffle_data(data, n_condition)

        rand_raw_non_conta_data = rand_data[:][0].reshape((no_epochs*1024,n_channels))
        rand_raw_conta_data = rand_data[:][1].reshape((no_epochs*1024,n_channels))

        # FUNCT1: Akta function hobe ja maps label generate kore quanitifier call korbe then 
        # duita condition er jonno quantifier diff ber kore return korbe hash map or list e
        randEffect = findEffect(rand_raw_conta_data, rand_raw_non_conta_data, optimalNumberOfCluster)
        randEffectSize[n_times] = randEffect
        
        n_times+=1

    # Function 2: While loop theke ber hobo then ei kaj korbo. Null hypothesis er porbability 
    # calculate korbe for each micrstate class and will return: Significantly different map+label and 
    # Significantly not different map+label
    
    sigDiffMapLabel, sigNotDiffMapLabel = findLabel(observedEffect, optimalNumberOfCluster, randEffectSize)

    return sigDiffMapLabel, sigNotDiffMapLabel