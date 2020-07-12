import Configuration
import EegArtifactAnalyzer
import PowerAnalysis
import ModifiedKmeans
import Cluster
import numpy as np
import matplotlib.pyplot as plt
from numpy import savetxt
import glob
import mne


# Function to merge subject data "condition: contaminated" wise
#def concatenateSubjectData(subjectConditionWiseData):
#	combinedData, countOfColumns, countChannels = [],[],[]
#	for key in subjectConditionWiseData:
#		combinedData.append(subjectConditionWiseData[key])
#		countOfColumns.append(subjectConditionWiseData[key].shape[1])
#		countChannels.append(subjectConditionWiseData[key].shape[0])
	
#	concatenatedData = np.zeros((len(combinedData),dtype='float')
#	for i in range(len(combinedData)):
#		concatenatedData[i] = combinedData[i].ravel('C')
	
#	concatenatedData = np.asarray(concatenatedData)
#	return  concatenatedData


#Channels location file reader function
def read_xyz(filename):
#    """Read EEG electrode locations in xyz format
#
#    Args:
#        filename: full path to the '.xyz' file
#    Returns:
#        locs: n_channels x 3 (numpy.array)
#    """
    ch_names = []
    locs = []
    with open(filename, 'r') as f:
        l = f.readline()  # header line
        while l:
            l = f.readline().strip().split("\t")
            if (l != ['']):
                ch_names.append(l[0])
                locs.append([float(l[1]), float(l[2]), float(l[3])])
            else:
                l = None
    return ch_names, np.array(locs)

def plotMicrostateMaps(maps, filename):
    channels, locs = read_xyz(filename)
    for i, map in enumerate(maps):
        #plt.figure(figsize=(2 * len(maps),5))
        plt.subplot(1, len(maps), i+1)
        plt.title("Maps: {}".format(i))
        mne.viz.plot_topomap(map, pos = locs[:, :2])

    return None


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
        finalEmgData2Raw = EegArtifactAnalyzer.formatForMicrostate(raw, finalEmgData2, list(set(ch_names_combined)))
        contaData = finalEmgData2Raw.get_data()
        subjectConditionWiseData.append(contaData)
        #subjectConditionWiseData['subject: ' + str(i)] = contaData 
        
        
        # Now we just form the other condition: "Non-Contaminated" simply with raw data
        # Loading the channels of interest
        channelsOptimalCluster = Configuration.channelList()
        # Loading only interested channels' data to reduce computation time
        data = raw.pick(picks = channelsOptimalCluster).get_data()
        subjectWiseData['subject: ' + str(i)] = data
        subjectData.append(data)
        i += 1

    subjectData = np.asarray(subjectData)
    #concatenateConditionWiseData = concatenateSubjectData(subjectConditionWiseData)
    subjectConditionWiseData = np.hstack((subjectConditionWiseData[0],subjectConditionWiseData[1],subjectConditionWiseData[2],subjectConditionWiseData[3]))
    noOfSamples = subjectConditionWiseData.shape[1] - (subjectConditionWiseData.shape[1] % subjectData.shape[0])
    timeSamples = noOfSamples//subjectData.shape[0]
    subjectConditionWiseData = subjectConditionWiseData[:,:noOfSamples].reshape(subjectData.shape[0],subjectData.shape[1], timeSamples)


    
    return subjectData, subjectConditionWiseData


# Function to conduct EEG microstate analysis on the  raw data for finding optimal number microstate classes or maps 
def analyzeMicrostate(trainDataPath):
    #if trainDataPath == '':
    #    trainDataPath = Configuration.defaultTrainDataFolder()
    
    subjectWiseData, subjectConditionWiseData = loadData(trainDataPath)
    #optimalCluster = Cluster.findOptimalCluster(subjectWiseData, subjectConditionWiseData)
    optimalCluster = 10
    data = subjectWiseData.mean(axis = 0).T
    # Zoom in of the data in to micro volt from volts
    data = data/1e-06
    # Removing the first time sample from all channels. The first sample is very close to zero. We can ignore that.
    data = data[1:,:]

    optimalMaps, labels, gfp_peaks, gev, cv = ModifiedKmeans.kmeans(data, n_maps=10, n_runs = 50, maxerr = 1e-6, 
                                                      maxiter = 1000, doplot = False)

    savetxt('optimalMaps.csv', optimalMaps, delimiter=',')
    filename = Configuration.channelLocationFile()
    plotMicrostateMaps(optimalMaps, filename)
    print('Stop')
    return optimalMaps, optimalCluster
    

   


   

     
   

   
    
       



    
    