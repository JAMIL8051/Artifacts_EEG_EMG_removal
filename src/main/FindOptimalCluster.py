import Configuration
import glob
import mne
import numpy as np
import math
import random
from scipy import signal


# This function loads the EEG files and do little preporcessing and store the data subject wise
def loadData():
# Later use Tkinter package to make user friendly
    filePath = "C:/Users/J_CHOWD/Desktop/EEG_microstate_analysis_papers/TestDataN-BackLucas/*.bdf"
    eegFiles = glob.glob(filePath)
    
    subjectWiseData = {}
    subjectData = []
# Loading the raw data in the dictionary
    i = 1
    for file in eegFiles:
        raw = mne.io.read_raw_bdf(file, preload=True)
        raw.filter(0.1, 100.0, fir_design='firwin')
        raw.notch_filter(np.arange(60, 241, 60), fir_design='firwin')
        # Selecting the EEG channels only
        raw.pick_types(meg = False, eeg=True, eog = False, stim = False)
        raw.set_eeg_reference('average', projection=False)
        #Setting the Bio Semi 64 channel montage
        raw.set_montage(Configuration.channelLayout())
        data = raw.get_data()
        subjectWiseData['subject: ' + str(i)] = data
        subjectData.append(data)
        i += 1
    
    subjectData = np.asarray(subjectData)
    n_ch = subjectData[0].shape[0]
    n_t = subjectData[0].shape[1]

    subjectData1 = np.zeros((len(subjectData), n_ch, n_t),dtype ='float')

    for i in range(len(subjectData)):
        subjectData1[i,:,:] =subjectData[i]
    
    return subjectData1

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

#     if instantaneous_eeg:
#         correlation = spatial_correlation(data, zero_mean(maps, 1).T, data.std(axis=1),
#                                                          maps.std(axis=1), data.shape[0])
#         correlation = correlation if polarity else abs(correlation)
#         label = np.argmax(correlation,axis=1)
#         return label

    gfp = data.std(axis=1)
    peaks, _ = signal.find_peaks(gfp, distance=distance, height=(gfp.mean() - n_std * gfp.std(), gfp.mean() + n_std * gfp.std()))
    correlation = spatial_correlation(data[peaks], zero_mean(maps, 1).T, gfp[peaks], maps.std(axis=1), data.shape[0])
    correlation = correlation if polarity else abs(correlation)
    return correlation 

def calcMeanCorrelation(testData, maps):
    meanCorrelation = fit_back(testData, maps, distance= 10, n_std=3, polarity=False, 
                               instantaneous_eeg = False)
    return meanCorrelation

def findOptimalCluster():
    subjectWiseData = loadData()
    allMaps = {}
    repetitions= {}
    # repetitions start here:

     # For loop ends for 250 times with randomly shuffling 4 subjects data with 50% train data and 50% test
    # data
    for i in range(250):
        shuffledSubjectWiseData = []
        randSubjects = random.sample(subjects,len(subjects))
        shuffledSubjectWiseData.append(subjectWiseData[randSubjects])

        shuffleData = np.asarray(shuffledSubjectWiseData)
        shuffleData = shuffleData.reshape(len(subjects),3,3)

        # 50% training data
        trainData = shuffleData[:shuffleData.shape[0]//2]

        # 50% test data
        testData = shuffleData[shuffleData.shape[0]//2:]

        meanTrainData = trainData.mean(axis = 0)
        meanTestData = testData.mean(axis = 0)

        n_maps = 3

        maxTotalGev = -1

        minCv = np.Infinity

        maxCorrelation = -1

        optimalCluster, optimalCluster1, optimalNumberOfCluster = -1, -1, -1

        while n_maps<21:    
            maps, labels, gfp_peaks, gev, cv = ModifiedKmeans.kmeans(meanTrainData, n_maps, n_runs = 50, maxerr = 1e-6, 
                                                      maxiter = 1000, doplot = False)

            totalGev = sum(gev)

            if totalGev > maxTotalGev:
                optimalCluster = n_maps
                maxTotalGev = totalGev

            if cv < minCv:
                optimalCluster1 = n_maps
                optimalMaps1.append(maps)
                minCv = cv

            meanCorrelation = calcMeanCorrelation(testData, maps)
            allMaps[str(n_maps)] = meanCorrelation 

            n_maps += 1

        repetitions[str(i)] = allMaps
    # End of for loop
   

    meanCorrelation = []

    for key in repetitions:
        for key1 in repetitions[key]:
            meanCorrelation.append(repetitions[key][key1])

    avgMeanCorrelation = {}

    n_maps = 3
    i = 0
    while n_maps < 21:
        temp = []
        temp.append(myList[250*i:250*(i+1)])
        avgMeanCorrelation[str(n_maps)] = temp
        n_maps += 1
        i += 1

    for key in avgMeanCorrelation:
        avgMeanCorrelation[key] = np.mean(np.asarray(avgMeanCorrelation[key]))

    optimalCluster = -1
    maxAvgMeanCorr = -1 

    for key in avgMeanCorrelation:
        if avgMeanCorrelation[key] > maxAvgMeanCorr:
            maxAvgMeanCorr = avgMeanCorrelation[key]
            optimalCluster = int(key)

    return subjectWiseData, optimalCluster
        
