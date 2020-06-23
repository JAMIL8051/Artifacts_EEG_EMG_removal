import numpy as np
import ModifiedKmeans
import Cluster 


def loadData(filePath):
# Later use Tkinter package to make user friendly
    eegFiles = glob.glob(filePath)
    
    subjectWiseData = {}
    subjectConditionWiseData = []
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

        # We have major two conditions as mentioned in Configuration file.
        # Starting to detect artifactual data segements to form the condition wise subject data. Condition = "Contaminated"
        artifactualData, finalEmgData2, ch_names_combined = PowerAnalysis.identifyArtifacts(raw)
        finalEmgData2Raw = EegArtifactAnalyzer.formatForMicrostate(raw, dataWithArtifactsDetected, list(set(ch_names_combined)))
        contaData = finalEmgData2Raw.get_data()
        subjectConditionWiseData.append(contaData)
        
        # Now we just form the other condition: "Non-Contaminated" simply with raw data
        channelsOptimalCluster = Configuration.channelList()
        data = raw.pick(picks = channelsOptimalCluster).get_data()
        subjectWiseData['subject: ' + str(i)] = data
        subjectData.append(data)
        i += 1
       
    
    subjectData = np.asarray(subjectData)
    subjectCondtionData = np.asarray(subjectConditionWiseData)

    subjectData1 = listToArray(subjectData)
    subjectConditionWiseData = listToArray(subjectCondtionData)
    
    return subjectData1, subjectConditionWiseData


# Function to conduct EEG microstate analysis on the  raw data for finding optimal number microstate classes or maps 
def analyzeMicrostate(trainDataPath):
    if trainDataPath == '':
        trainDataPath = Configuration.defaultTrainDataFolder()
    
    subjectWiseData, subjectConditionWiseData = loadData(trainDataPath)
    subjectWiseData, subjectConditionWiseData, optimalCluster = Cluster.findOptimalCluster()

    data = subjectWiseData.mean(axis = 0).T
    # Zoom in of the data in to micro volt from volts
    data = data/1e-06
    # Removing the first time sample from all channels. The first sample is very close to zero. We can ignore that.
    data = data[1:,:]

    optimalMaps, labels, gfp_peaks, gev, cv = ModifiedKmeans.kmeans(data, optimalCluster, n_runs = 10, maxerr = 1e-6, 
                                                      maxiter = 1000, doplot = False)
    return optimalMaps, optimalCluster
    

   


   

     
   

   
    
       



    
    