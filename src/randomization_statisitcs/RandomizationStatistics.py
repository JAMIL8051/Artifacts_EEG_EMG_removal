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
def findEffect(conta_data, non_conta_data, opt_cluster1, optimalNumberOfCluster):
    n_maps = opt_cluster1 
    conta_maps, conta_labels = ModifiedKmeans.kmeans(conta_data, n_maps, n_runs= 10, 
                                                     maxerr=1e-6, maxiter=1000, doplot = False)
    conta_maps_1, conta_labels_1 = ModifiedKmeans.kmeans(conta_data, n_maps = optimalNumberOfCluster, 
                                                         n_runs= 10, maxerr=1e-6, maxiter=1000, 
                                                         doplot = False)
    
    non_conta_maps, non_conta_labels = ModifiedKmeans.kmeans(non_conta_data, n_maps, n_runs= 10, 
                                                             maxerr=1e-6, maxiter=1000, 
                                                             doplot = False)
    non_conta_maps_1, non_conta_labels_1 = ModifiedKmeans.kmeans(non_conta_data, 
                                            n_maps = optimalNumberOfCluster, n_runs= 10, 
                                            maxerr = 1e-6, maxiter = 1000, doplot = False)
    
    # Calculatin the microstate qunatifiers
    condition_0_0 = MicrostateQuantifiers.quantifyMicrostates(conta_maps, conta_labels)
    condition_0_1 = MicrostateQuantifiers.quantifyMicrostates(conta_maps_1, conta_labels_1)
    condition_1_0 = MicrostateQuantifiers.quantifyMicrostates(non_conta_maps, non_conta_labels)
    condition_1_1 = MicrostateQuantifiers.quantifyMicrostates(non_conta_maps_1, non_conta_labels_1)
    
    # Choosing only one qunatifier for simplification
    val=[]

    for key in condition_0_0['Count of time points']:
        val.append(condition_0_0['countOfTimePoints'][key])

    val1 = []
    for key in condition_1_0['Count of time points']:
        val1.append(condition_1_0['countOfTimePoints'][key])

    val, val1 = np.asarray(val), np.asarray(val1) 
    effectCvTech = val-val1

    val2=[]

    for key in condition_0_1['Count of time points']:
        val.append(condition_0_1['countOfTimePoints'][key])

    val3 = []
    for key in condition_1_1['Count of time points']:
        val1.append(condition_1_1['countOfTimePoints'][key])

    val2, val3 = np.asarray(val2), np.asarray(val3) 
    effectWithModelTech = val-val1
    return effectCvTech, effectWithModelTech


#Small twick to find the observed effect
def findObservedEffect(raw, rawWithArtifactsDetected, opt_cluster1, optimalNumberOfCluster):
    conta_data = rawWithArtifactsDetected.get_data().T
    non_conta_data = raw.pick(picks = rawWithArtifactsDetected.ch_names).get_data().T
    
    observedEffectCv, observedEffectModel  = findEffect(conta_data, non_conta_data, opt_cluster1, optimalNumberOfCluster)
    

    return observedEffectCv, observedEffectModel


# Finally finds the label by calculating the probability using no. of random shffles chosen
def findLabel(opt_cluster1, optimalNumberOfCluster, randEffectSize, randEffectSize1):
    microstateClass = np.zeros((opt_cluster1),dtype = 'int')
    
    
    for j in range(opt_cluster1):
        count = 0
        for i in range(randEffectSize.shape[0]):
            if randEffectSize[i][j] >= observedEffectCv[j]:
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
    
    microstateClass1 = np.zeros((optimalNumberOfCluster), dtype = 'int')
    
    for j in range(optimalNumberOfCluster):
        count = 0
        for i in range(randEffectSize1.shape[0]):
            if randEffectSize1[i][j] >= observedEffectModel[j]:
                count += 1
    
        microstateClass1[j] = count
    
    microstateClassProbability1 = microstateClass1/randEffectSize1.shape[0]
    
    sigDiffMapLabel1 = []
    sigNotDiffMapLabel1 = []
    for i in range(len(microstateClassProbability1)):
        if microstateClassProbability1[i] <=0.05:
            sigDiffMapLabel1.append(i)
        else:
            sigNotDiffMapLabel1.append(i)
    return sigDiffMapLabel, sigDiffMapLabel1, sigNotDiffMapLabel, sigNotDiffMapLabel1


# We randomly assign a condition on each epochs for single subject analysis: 
# Future target on subject basis data: Randomly assign condition on to subject data: See:
"""A Tutorial on Data-Driven Methods for Statistically Assessing ERP Topographies by Thomas Koenig,
    Maria Stein, Matthias Grieder, Mara Kottlow for more details.
"""
# Final implementation of randomization statistics to get the map labels
def randomizationStat(raw, rawWithArtifactsDetected, 
											artifactualData, 
											opt_cluster1, 
											optimalNumberOfCluster):
    #FUNCT1:Call hobe observed effect ber korar jonno: 
    #2 times: 1 for opt_cluster1 1 for optimalNumberOfCluster 
    #chinta korbo: maps plus label duitai return koriye dictionary te store korbe:
    #ei gula hobe dictionary: sigDiffMapLabel, sigNotDiffMapLabel
    observedEffectCv, observedEffectModel = findObservedEffect(raw, rawWithArtifactsDetected, 
                                                               opt_cluster1,
                                        optimalNumberOfCluster)
    

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
    randEffectSize = np.zeros((1000, opt_cluster1),dtype = 'float')
    randEffectSize1 = np.zeros((1000, optimalNumberOfCluster),dtype = 'float')
    while n_times<1000:
    
        rand_data = shuffle_data(data, n_condition)

        rand_raw_non_conta_data = rand_data[:][0].reshape((no_epochs*1024,n_channels))
        rand_raw_conta_data = rand_data[:][1].reshape((no_epochs*1024,n_channels))

        # FUNCT1: Akta function hobe ja maps label generate kore quanitifier call korbe then 
        # duita condition er jonno quantifier diff ber kore return korbe hash map or list e
        randEffectCv, randEffectModel  = findEffect(rand_raw_conta_data, rand_raw_non_conta_data, 
                                                    opt_cluster1, optimalNumberOfCluster)
        randEffectSize[n_times] = randEffectCv
        randEffectSize1[n_times] = randEffectModel

        n_times+=1
# Function 2: While loop theke ber hobo then ei kaj korbo. Null hypothesis er porbability 
# calculate korbe for each micrstate class and will return: Significantly different map+label and 
# Significantly not different map+label
    
    sigDiffMapLabel, sigDiffMapLabel1, sigNotDiffMapLabel, sigNotDiffMapLabel1 = findLabel(opt_cluster1, optimalNumberOfCluster, randEffectSize, randEffectSize1)

    return sigDiffMapLabel, sigNotDiffMapLabel, sigDiffMapLabel1, sigNotDiffMapLabel1