import os
import os.path
from src.main import Configuration
import numpy as np
import ModifiedKmeans



# The function segement the data in to test and train data for micro state analysis
def segmentData(raw):
    channelNamesMap, region = Configuration.channels()
    ch_list = []

    for key in channelNamesMap:
        if key!='Fz':
            ch_list.append(channelNamesMap[key])

    channels = sum(ch_list,[])

    data, times = raw.pick(picks = channels).get_data(return_times = True).T

    # Creating 50% training  and test data
    trainData = data[:len(times)//2,:]
    testData = data[len(times)//2:,:]

    return data, trainData, testData, channelNamesMap, channels


# Calculating the correaltion of microstate models with test data
def calcMeanCorrelation(testData, trainData, n_maps):
    
    meanCorrelationList = []

    for i in range(ConfigFile.repetitonsCount()):
        randomMaps = ModifiedKmeans.kmeans(trainData, n_maps, n_runs = 10, maxerr = 1e-6, 
                                          maxiter = 1000, doplot = False)
        correlation = ((np.cov(testData,randomMaps))/(np.var(testData)*np.var(randomMaps)))
        meanCorrelation = np.mean(correlation)
        meanCorrelationList.append(meanCorrelation)
        avgMeanCorrelation = np.mean(np.array(meanCorrelationList))
    
    return avgMeanCorrelation


# Function to find the optimal number of clusters
def findOptimalCluster(data, trainData, testData):
    n_maps = 3
    optimalMaps1 = []
    optimalModelMaps =[]
    meanCorrelation ={}
    maxTotalGev = -1
    optimalCluster = -1
    optimalCluster1 = -1
    minCv = np.Infinity
    avgMeanCorrelationList = []
    maxCorrelation = -1
    while n_maps <= ConfigFile.numberOfCluster():
        # Process with finding optimal number cluster using gev and cv concept
        maps, labels, gfp_peaks, gev, cv = ModifiedKmeans.kmeans(data, n_maps, n_runs = 10, maxerr = 1e-6, 
                                              maxiter = 1000, doplot = False)
        totalGev = sum(gev)

        if totalGev > maxTotalGev:
            optimalCluster = n_maps
            #optimalMaps.append(maps)
            maxTotalGev = totalGev

        if cv < min_cv:
            optimalCluster1 = n_maps
            optimalMaps1.append(maps)
            minCv = cv 
       
        # Method for selection of optimal microstate model with test data parameter
        avgMeanCorrelation = calcMeanCorrelation(testData, trainData, n_maps)
        
        if avgMeanCorrelation > maxCorrelation:
            optimalNumberOfCluster = n_maps
            optimalModelMaps.append(maps)
            maxCorrelation = avgMeanCorrelation
            
        n_maps += 1
        optimalMaps1 = np.asarray(optimalMaps1).reshape(optimalCluster1,data.shape[1])
        optimalModelMaps = np.asarray(optimalModelMaps).reshape(optimalNumberOfCluster,
                                                                data.shape[1])
      
    return opt_cluster1, optimalMaps1, optimalNumberOfCluster, optimalModelMaps


# Function to conduct EEG microstate analysis on the  raw data for finding optimal number microstate
# class or map 
def analyzeMicrsotate(raw):
    #artifactualData pore shorabo ei parameter

    #Setting the Bio Semi 64 channel montage
    raw.set_montage(Configuration.channelLayout())

    #Selecting the EEG channels only
    raw.pick_types(meg=False, eeg=True, eog = False, stim = False)
    
    # Data segmentation
    data, trainData, testData, channelNamesMap, channels = segmentData(raw)
    # Determining the optimal number of clusters from the data of the EMG artifact prone regions
    opt_cluster1, optimalNumberOfCluster = findOptimalCluster(data, trainData, testData)
    return opt_cluster1, optimalNumberOfCluster


   


   

     
   

   
    
       



    
    