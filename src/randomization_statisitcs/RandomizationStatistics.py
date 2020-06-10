import mne
import numpy as np


# Function to shuffle the data: We randomly the condition on to  epochs for single subject: 
# Future target on subject basis data: Randomly assign condition on to subject data: See:
"""A Tutorial on Data-Driven Methods for Statistically Assessing ERP Topographies by Thomas Koenig,
    Maria Stein • Matthias Grieder, Mara Kottlow for more details.
"""
def shuffle_data(data, n_condition):
    for i in range(data.shape[0]):
        random_index = np.random.permutation(n_condition)
        data[i] = data[i][random_index]
    return data



# Function to conduct the random instances and generate the distribution under the null hypothesis
def randomizationStat(raw, dataWithArtifactsDetectedRaw, 
											artifactualData, 
											opt_cluster1, 
											optimalNumberOfCluster):

    #FUNCT1:Call hobe observed effect ber korar jonno: 
    #2 times: 1 for opt_cluster1 1 for optimalNumberOfCluster 
    #chinta korbo: maps plus label duitai return koriye dictionary te store korbe:
    #ei gula hobe dictionary: sigDiffMapLabel, sigNotDiffMapLabel



    channels = dataWithArtifactsDetectedRaw.ch_names
    n_channels = len(channels)

    raw_conta_data, times = dataWithArtifactsDetectedRaw.get_data(return_times = True)
    

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
    n_condition = Configuration.condition()
    data = np.concatenate((raw_non_conta_data,raw_conta_data), axis = 1)
    
    n_times = 0
    randEffectSize = np.zeros((optimalNumberOfCluster,1000,1),dtype = 'float')
    randEffectSize1 = np.zeros((opt_cluster1,1000,1),dtype = 'float')
    while n_times<1000:
    
        rand_data = shuffle_data(data, n_condition)
        #half_way = len(no_epoch*time_samples)//2
        rand_raw_non_conta_data = rand_data[:][0].reshape((no_epochs*1024,n_channels))
    
        rand_raw_conta_data = rand_data[:][1].reshape((no_epochs*1024,n_channels))
        # FUNCT1: Akta function hobe ja maps label generate kore quanitifier call korbe then 
        # duita condition er jonno quantifier diff ber kore return korbe hash map or list e

    # Function 2: While loop theke ber hobo then ei kaj korbo 
    # Null hypothesis er porbability calculate korbe for each micrstate and will return
    # Significantly different map+label and Significantly not different map+label   
        n_times+=1

    return sigDiffMapLabel, sigNotDiffMapLabel