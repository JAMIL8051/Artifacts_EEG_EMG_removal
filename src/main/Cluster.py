import Configuration
import EegArtifactAnalyzer
import ModifiedKmeans
import glob
import mne
import numpy as np
import math
import random
from scipy import signal


#Function to calculate the mean along the n_ch axis that along the rows: Courtesy Wenjun Jia
def zero_mean(data, axis=0):
    mean = data.mean(axis=1, keepdims=True)# keep dimension parameter preserves the original dimension after averaging
    return data - mean #We can subtract as we preserved the dimension


#Function for spataial correlation: Courtesy Wenjun Jia
def spatial_correlation(averaged_v1, averaged_v2, std_v1, std_v2, n_ch):
    correlation = np.dot(averaged_v1, averaged_v2) / (n_ch * np.outer(std_v1, std_v2))
    return correlation


# Function to find the spatial correlation of the model maps with the test data

def fit_back(data, maps, distance= 10, n_std=3, polarity=False, instantaneous_eeg = True):

    if instantaneous_eeg:
        correlation = spatial_correlation(data, zero_mean(maps, 1).T, data.std(axis=1),
                                                        maps.std(axis=1), data.shape[0])

        correlation = correlation if polarity else abs(correlation)
       

    gfp = data.std(axis=1)
    peaks, _ = signal.find_peaks(gfp, distance=distance, height=(gfp.mean() - n_std * gfp.std(), gfp.mean() + n_std * gfp.std()))
    correlation = spatial_correlation(data[peaks], zero_mean(maps, 1).T, gfp[peaks], maps.std(axis=1), data.shape[0])
    correlation = correlation if polarity else abs(correlation)

    return correlation 


def calcMeanCorrelation(testData, maps):
    correlation = fit_back(testData, maps, distance= 10, n_std=3, polarity=False, 
                               instantaneous_eeg = True)
    meanCorrelation = correlation.mean()
    return meanCorrelation


def meanCorrelation(subjectWiseData, subjects):
    randSubjects = random.sample(subjects,len(subjects))
   
    
    shuffleData = subjectWiseData[randSubjects] 
    
    
    # 50% training data
    trainData = shuffleData[:shuffleData.shape[0]//2]
    
    # 50% test data
    testData = shuffleData[shuffleData.shape[0]//2:]
    
    meanTrainData = trainData.mean(axis = 0)
    # Just unit conversion: From volt to microvolt
    meanTrainData = meanTrainData/1e-6
    meanTrainData = meanTrainData - meanTrainData.mean(axis = 0, keepdims = True)
    meanTrainData = meanTrainData[1:,:]
    meanTestData = testData.mean(axis = 0)
    meanTestData = meanTestData/1e-6
    
    n_maps = 3
    
    meanCorrelation = np.zeros((Configuration.numberOfCluster()-3),dtype = 'float')

    while n_maps<Configuration.numberOfCluster():    
        maps, labels, gfp_peaks, gev, cv = ModifiedKmeans.kmeans(meanTrainData.T, n_maps, n_runs = 50, maxerr = 1e-6, 
                                                  maxiter = 1000, doplot = False)
          
        meanCorrelation[n_maps-3] = calcMeanCorrelation(meanTestData, maps)
        
        n_maps += 1

    return meanCorrelation
        

def findOptimalCluster(subjectWiseData, subjectConditionWiseData):
    subjects = list(range(subjectWiseData.shape[0]))
    
    # We always start from the number of clusters = 3. So 3 is deducted to maintain the range. 
    meanCorrelations = np.zeros((Configuration.repetitionsCount(),(Configuration.numberOfCluster()-3)), dtype='float')
    
    # For loop ends for 250 times with randomly shuffling 4 subjects data with 50% train data and 50% test
    # data
    for i in range(Configuration.repetitionsCount()):
        shuffledSubjectWiseData = []

        meanCorrelationSubjectWise = meanCorrelation(subjectWiseData, subjects) 
        meanCorrelationCondtionWise = meanCorrelation(subjectConditionWiseData, subjects) 

        resultantCorrelation = (meanCorrelationSubjectWise + meanCorrelationCondtionWise)/2
        meanCorrelations[i] = resultantCorrelation
        
                
    avgMeanCorrelation = meanCorrelations.mean(axis = 0)
    
    optimalCluster = 0
    # Here the index 0,1  represent cluster no. 3,4 s respectively and so on. As a result, 3 has been added again. 
    for i in range(len(avgMeanCorrelation)):
        if avgMeanCorrelation[i] == max(avgMeanCorrelation):
            optimalCluster = i + 3

    print('Check done')

    return optimalCluster
    

    #meanCorrelation = []

    #for key in allMaps:
    #    for key1 in repetitions:
    #        meanCorrelation.append(repetitions[key1][key])

    #avgMeanCorrelation = {}

    #n_maps = 3
    #i = 0
    #j = Configuration.repetitionsCount()
    #while n_maps < Configuration.numberOfCluster():
    #    temp = []
    #    temp.append(meanCorrelation[j*i:j*(i+1)])
    #    avgMeanCorrelation[str(n_maps)] = temp
    #    n_maps += 1
    #    i += 1

    #for key in avgMeanCorrelation:
    #    avgMeanCorrelation[key] = np.mean(np.asarray(avgMeanCorrelation[key]))

    #optimalCluster = -1
    #maxAvgMeanCorr = -1 

    #for key in avgMeanCorrelation:
    #    if avgMeanCorrelation[key] > maxAvgMeanCorr:
    #        maxAvgMeanCorr = avgMeanCorrelation[key]
    #        optimalCluster = int(key)

    #repetitions= {}
    # repetitions start here:

