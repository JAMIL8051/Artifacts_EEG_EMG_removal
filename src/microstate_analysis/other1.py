# The function segement the data in to test and train data for micro state analysis
def segmentData(raw):

    channelsOptimalCluster = Configuration.channelList() 
    raw = raw.pick(picks = channelsOptimalCluster)
    
    data, times = raw.get_data(return_times = True)
    data = data.T

    # Creating 50% training  and test data
    trainData = data[:len(times)//2,:]
    testData = data[len(times)//2:,:]

    return data, trainData, testData


# Calculating the correaltion of microstate models with test data

#Function to calculate the mean along the n_ch axis that along the rows: Courtesy Wenjun Jia
def zero_mean(data, axis=0):
    mean = data.mean(axis=1, keepdims=True)# keep dimension parameter preserves the original dimension after averaging
    return data - mean #We can subtract as we preserved the dimension


#Function for spataial correlation: Courtesy Wenjun Jia
def spatial_correlation(averaged_v1, averaged_v2, std_v1, std_v2, n_ch):
    correlation = np.dot(averaged_v1, averaged_v2) / (n_ch * np.outer(std_v1, std_v2))
    return correlation


# Function to find the spatial correlation of the model maps with the test data
def fit_back(data, maps, distance= 10, n_std=3, polarity=False, instantaneous_eeg = False):

    if instantaneous_eeg:
        correlation = spatial_correlation(data, zero_mean(maps, 1).T, data.std(axis=1),
                                                        maps.std(axis=1), data.shape[0])
        correlation = correlation if polarity else abs(correlation)
         
        return correlation

    gfp = data.std(axis=1)
    peaks, _ = signal.find_peaks(gfp, distance=distance, height=(gfp.mean() - n_std * gfp.std(), gfp.mean() + n_std * gfp.std()))
    correlation = spatial_correlation(data[peaks], zero_mean(maps, 1).T, gfp[peaks], maps.std(axis=1), data.shape[0]) 
    correlation = correlation if polarity else abs(correlation)
    return correlation



def calcMeanCorrelation(testData, trainData, n_maps):
    
    meanCorrelationList = []

    for i in range(Configuration.repetitonsCount()):
        randomMaps, randomLabels, randomGfp_peaks, randomGev, randomCv = ModifiedKmeans.kmeans(
            trainData, n_maps, n_runs = 10, maxerr = 1e-6, maxiter = 1000, doplot = False)
        correlation = fit_back(testData, randomMaps, distance= 10, n_std=3, polarity=False, 
                               instantaneous_eeg = False)

        #correlation = ((np.cov(testData, randomMaps))/(np.var(testData)*np.var(randomMaps)))
        meanCorrelationList.append(correlation.mean())

    avgMeanCorrelation = np.mean(np.array(meanCorrelationList))
    
    return avgMeanCorrelation


# Function to find the optimal number of clusters
def findOptimalCluster(data, trainData, testData):
    n_maps = 3
    optimalMaps1 = []
    optimalNumberMaps =[]

    maxTotalGev = -1

    optimalCluster = -1
    optimalCluster1 = -1
    optimalNumberOfCluster = -1
    
    minCv = np.Infinity

    maxCorrelation = -1

    while n_maps < Configuration.numberOfCluster():
        # Process with finding optimal number cluster using gev and cv concept
        maps, labels, gfp_peaks, gev, cv = ModifiedKmeans.kmeans(trainData, n_maps, n_runs = 50, maxerr = 1e-6, 
                                              maxiter = 1000, doplot = False)
        totalGev = sum(gev)

        if totalGev > maxTotalGev:
            optimalCluster = n_maps
            maxTotalGev = totalGev

        if cv < minCv:
            optimalCluster1 = n_maps
            optimalMaps1.append(maps)
            minCv = cv 
       
        # Method for selection of optimal microstate model with test data parameter
        avgMeanCorrelation = calcMeanCorrelation(testData, trainData, n_maps)
        
        if avgMeanCorrelation > maxCorrelation:
            optimalNumberOfCluster = n_maps
            optimalNumberMaps.append(maps)
            maxCorrelation = avgMeanCorrelation
 
        n_maps += 1

    optimalMaps1 = np.asarray(optimalMaps1).reshape(optimalCluster1, data.shape[1])
    optimalNumberMaps = np.asarray(optimalNumberMaps).reshape(optimalNumberOfCluster, data.shape[1]) 
        
      
    return optimalCluster1, optimalMaps1, optimalNumberOfCluster, optimalNumberMaps




#def analyzeMicrostate(raw):
   

    #Setting the Bio Semi 64 channel montage
    #raw.set_montage(Configuration.channelLayout())

    #Selecting the EEG channels only
    #raw.pick_types(meg=False, eeg=True, eog = False, stim = False)
    
    # Data segmentation
    #data, trainData, testData = segmentData(raw)
    # Determining the optimal number of clusters from the raw data 
    #opt_cluster1, optimalMaps1, optimalNumberOfCluster, optimalNumberMaps = findOptimalCluster(data, trainData, testData)



#return opt_cluster1, optimalMaps1, optimalNumberOfCluster, optimalNumberMaps 
