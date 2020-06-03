import os
import os.path
from src.main import Configuration
import numpy as np
import ModifiedKmeans as kmeans
import MicrostateMapsAnalysis
import microstates
import stat

# The function segement the data in to test and train data for micro state analysis
def segmentData(raw):
    channelNamesMap, region = Configuration.channels()
    ch_list = []

    for key in channelNamesMap:
        if key!='Fz':
            ch_list.append(channelNamesMap[key])

    channels = sum(ch_list,[])

    data, times = raw.pick(picks = channels).get_data(return_times = True).T
    # Creating 50% training data
    trainData = data[:len(times)//2,:]
    testData = data[len(times)//2:,:]

    return trainData, testData, channelNamesMap, channels


# Function to find the optimal number of clusters
def calcMeanCorrelation(testData, trainData, n_maps):
    
    meanCorrelationList = []

    for i in range(ConfigFile.repetitonsCount()):
        randomMaps = kmeans(trainData, n_maps, n_runs = 10, maxerr = 1e-6, 
                                          maxiter = 1000, doplot = False)
        correlation = ((np.cov(testData,randomMaps))/(np.var(testData)*np.var(randomMaps)))
        meanCorrelation = np.mean(correlation)
        meanCorrelationList.append(meanCorrelation)
        avgMeanCorrelation = np.mean(np.array(meanCorrelationList))
    
    return avgMeanCorrelation

def findOptimalCluster(trainData, testData):
    n_maps = 3
    meanCorrelation ={}
    maxTotalGev = -1
    optimalCluster = -1
    optimalCluster1 = -1
    minCv = np.Infinity
    avgMeanCorrelationList = []
    maxCorrelation = -1
    while n_maps <= ConfigFile.numberOfCluster():
        # Process with finding optimal number cluster using gev and cv concept
        maps, labels, gfp_peaks, gev, cv = kmeans(trainData, n_maps, n_runs = 10, maxerr = 1e-6, 
                                              maxiter = 1000, doplot = False)
        totalGev = sum(gev)

        if totalGev > maxTotalGev:
            optimalCluster = n_maps
            maxTotalGev = totalGev

        if cv < min_cv:
            optimalCluster1 = n_maps
            minCv = cv 
       
        # Method for selection of optimal microstate model with test data parameter
        avgMeanCorrelation = calcMeanCorrelation(testData, trainData, n_maps)
        
        if avgMeanCorrelation > maxCorrelation:
            optimalNumberOfCluster = n_maps
            maxCorrelation = avgMeanCorrelation
            
        n_maps += 1
      
    return opt_cluster1, optimalNumberOfCluster

# Function to analyze the data on EEG microstate analysis
def analyzeMicrsotate(raw, artifactualData):
    #Setting the Bio Semi 64 channel montage
    raw.set_montage('biosemi64')

    #Selecting the EEG channels only
    raw.pick_types(meg=False, eeg=True, eog = False, stim = False)
    
    # Data segmentation
    trainData, testData, channelNamesMap, channels = segmentData(raw)
    # Determining the optimal number of clusters from the data of the EMG artifact prone regions
    opt_cluster, opt_cluster1, optimalNumberOfCluster = findOptimalCluster(trainData, testData)

     # Main_part: Region wise microstate analysis: Both contaminated data and data for testing contamination
    # Microstate analysis on the detected artifactual data epochs on the basis of regions using CV technique
    # also optimal number of microstate model technique 
  
    mapsRegion = {}
    modelMapsRegion = {}
    labelsRegion = {}
    modelLabelsRegion = {}
    for key in artifactualData:
        
        maps, labels, gfp_peaks, gev, cv = kmeans(artifactualData[key], n_maps = opt_cluster1, n_runs = 10, 
                                                  maxerr = 1e-6, maxiter = 1000, doplot = False )
        modelMaps, modelLabels, modelGfp_peaks, modelGev, modelCv = kmeans(artifactualData[key], 
                                                                           n_maps = optimalNumberOfCluster, n_runs = 10, 
                                                                           maxerr = 1e-6, 
                                                                           maxiter = 1000, doplot = False)
        mapsRegion[key] = maps
        modelMapsRegion[key] = modelMaps

        labelsRegion[key] = labels
        modelLabelsRegion[key] = modelLabels
    
        

    # Application on the primary electrodes:
    primaryChannels =[]
    for key in channelNamesMap:
        if key!='Fz':
            primaryChannels.append(key)
    primaryData = raw.pick_channels(ch_names = primaryChannels).get_data()
     # Cv technique
    primaryMaps,primaryLabels =kmeans(primaryData, n_maps = opt_cluster1, n_runs = 10, maxerr = 1e-6, 
                                        maxiter = 1000, doplot = False)

    # Micorstate model technique
    modelPrimaryMaps, modelPrimaryLabels = kmeans(primaryData, n_maps = optimalNumberOfCluster,n_runs = 10, 
                                                  maxerr = 1e-6, maxiter = 1000, doplot = False)
   
    # Application microstate analysis on the combined channels(20 channels in total): 
    # Test data from preprocessed raw data using: 
    
    # Cv technique
    combinedData = raw.pick_channels(ch_names = channels).get_data()
    mapsCombined, labelsCombined, gfp_peaks, gev, cv = kmeans(combinedData, n_maps = opt_cluster1, 
                                                            n_runs = 10, maxerr = 1e-6, 
                                        maxiter = 1000, doplot = False)

    # Micorstate model technique
    modelMapsCombined, modelLabelsCombined, modelGfp_peaks, modelGev, modelCv = kmeans(combinedData, 
                                                                    n_maps = optimalNumberOfCluster, 
                                                                    n_runs = 10, 
                                                                    maxerr = 1e-6, 
                                                                    maxiter = 1000, doplot = False)
    # Pore dekhbo
    return None
            
    #return mapsRegion, modelMapsRegion, labelsRegion, modelLabelsRegion, mapsCombined, labelsCombined, modelMapsCombined, modelLabelsCombined   
    


    #1. Application of microstate analysis on test Data: 20 channels combined data with optimal cluster (3)
    # Using the GEV and CV technique for finding the optimal number of clusters on the test data
    #opt_maps, opt_labels, opt_gfp_peaks, opt_gev, opt_cv = kmeans(testData, n_maps = opt_cluster, 
    #                                                              n_runs = 10, maxerr = 1e-6, 
    #                                          maxiter = 1000, doplot = False)

    #opt_maps1, opt_labels1, opt_gfp_peaks1, opt_gev1, opt_cv1 = kmeans(testData, n_maps = opt_cluster1, 
    #                                                                   n_runs = 10, maxerr = 1e-6, 
    #                                                                maxiter = 1000, doplot = False)

    #modelMaps, modelLabels, modelGfp_peaks, modelGev, modelCv = kmeans(testData, n_maps = optimalNumberOfCluster, 
    #                                                              n_runs = 10, maxerr = 1e-6, 
    #                                          maxiter = 1000, doplot = False)

   
    ##3. Application of another microstate analysis function to segment the test(non detected data) 20 channels 
    ## combined data and generate the microstate classes
    
    #channels = sum(ch_list,[])
    #data = raw.copy().pick_channels(ch_names = channels).get_data()
    
    ## Segment the data into optimal no of  microstates
    #maps_opt, segmentation_opt = microstates.segment(data, n_states= opt_cluster, max_n_peaks=10000000000,
    #                                                max_iter=5000, normalize=True)
    #maps_opt1, segmentation_opt1 = microstates.segment(data, n_states= opt_cluster1, 
    #                                                   max_n_peaks=10000000000, max_iter=5000,
    #                                                   normalize=True)








    #4. Application of micrsotate analysis on the rest of the channels
    #Formation of the residue data from the raw object produced by the rest of the EEG channels(64-20)
    # of other brain regions

    #residueData = raw.copy().drop_channels(ch_names = channels).get_data()[:len(channels),:]

    ## Application of microstate analysis on first method.
    #opt_residue_maps, opt_residue_labels, opt_residue_gfp_peaks, opt_residue_gev, opt_residue_cv = kmeans(
    #                                                        residueData.T, n_maps = opt_cluster, n_runs = 10, 
				#									maxerr = 1e-6, maxiter = 1000, doplot = False)
    #opt_residue_maps1, opt_residue_labels1, opt_residue_gfp_peaks1, opt_residue_gev1, opt_residue_cv1 = kmeans(
    #                                                            residueData.T, n_maps = opt_cluster1, n_runs = 10, 													maxerr = 1e-6, 
    #                                                            maxiter = 1000, doplot = False)
    
    

    
    ## Second method to apply the microstate analysis
    #mapsResidue, segmentationResidue = microstates.segment(residueData, n_states= opt_cluster,
    #                                                         max_n_peaks=10000000000, 
    #                                                         max_iter=5000, normalize=True)
    #mapsResidue1, segmentationResidue1 = microstates.segment(residueData, n_states= opt_cluster1,
    #                                                         max_n_peaks=10000000000, 
    #                                                         max_iter=5000, normalize=True)


   

     
   

   
    
       



    
    