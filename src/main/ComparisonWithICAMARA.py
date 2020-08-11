import Configuration
from tkinter import filedialog
import tkinter as tk
import mne
import numpy as np
from pyprep.noisy import Noisydata


def get_raw_data_file_path():
    root = tk.Tk()
    root.withdraw()
    data_file_path  = filedialog.askopenfilename()
    return data_file_path


#Loading the raw data
def load_raw_data(filepath):
    raw = mne.io.read_raw_eeglab(filepath, preload = True) 
    return raw

# Main function to load
def preprocess_raw_data():
    filepath = get_raw_data_file_path()    
    raw = load_raw_data(filepath)
    raw.set_montage('biosemi64')
    
    return raw

def calculateQualtiyMeasures(data,thresholdValues):
    # data is a numpy array obtained from MNE raw object using get_data() method
    # thresholdValues is a list containing the threshold values 10, 20, 30, 40, 50, 60,70, 80,90 
    # micro-volts
    
    data = data/1e-06 # converting to micro-volts as thresholdValues will be considered in micro-volts
    
    flat_data = abs(data.ravel())
    
    
    overallHighAmplitude = []
    timeHighVarience = []
    channelHighVarience = []
    
    
    valTime = data.std(axis = 0, keepdims =True)
    valChannel = data.std(axis = 1,keepdims = True)
    
    for k in range(len(thresholdValues)):
        summation = 0
        summationTimePoints = 0
        summationChannels = 0
        
        for i in range(flat_data.shape[0]):
            if flat_data[i]>thresholdValues[k]:
                summation = summation + flat_data[i]
        overallHighAmplitude.append(summation/flat_data.shape[0]) 
        
        for j in range(data.shape[1]):
            if valTime[0,j]>thresholdValues[k]:
                summationTimePoints = summationTimePoints + valTime[0,j] 
         
        timeHighVarience.append(summationTimePoints/data.shape[1])
                
        for i in range(data.shape[0]):
            if valChannel[i,0]> thresholdValues[k]:
                summationChannels = summationChannels + valChannel[i,0]  
                
        channelHighVarience.append(summationChannels/data.shape[0])
                    
    return overallHighAmplitude, timeHighVarience, channelHighVarience



def compareWithIcaMara(finalEmgRaw, visualize = False):
    print('Give the artifacts free file in set format obtained using ICA_MARA in MATLAB')
    raw = preprocess_raw_data()
    # Loading the interested channels only
    channelsOptimalCluster = Configuration.channelList()
    
    rawICAMARA = raw.pick(picks = channelsOptimalCluster)

    if visualize:
        plt.figure()
        ax = plt.axes()
        finalEmgRaw.plot_psd(fmin=45.0, fmax= 70.0, tmin=0.0, tmax=200.0, proj=False, n_fft= 512*2, ax =ax, 
                             n_overlap= 0, picks = channelsOptimalCluster , show =False, average = False, 
                             xscale='linear',dB=False, estimate='amplitude')
        ax.set_title(Configuration.setAsdPlotTittle())

        plt.figure()
        ax1 = plt.axes()
        rawICAMARA.plot_psd(fmin=45.0, fmax= 70.0, tmin=0.0, tmax=200.0, proj=False, n_fft= 512*2, ax =ax1, 
                                 n_overlap= 0, picks = channelsOptimalCluster , show =False, average = False, 
                                 xscale='linear',dB=False, estimate='amplitude')
        ax1.set_title(Configuration.setAsdPlotTittle())


    # Comparsion with PREP analysis:
    nd = Noisydata(rawICAMARA)
    nd.find_all_bads(ransac = False)
    bads = nd.get_bads(verbose=True)
    print('PREP analysis results of artifact free data obtained from ICA_MARA')

    # Comparison with PREP analysis for the proposed method data
    nd = Noisydata(finalEmgRaw)
    nd.find_all_bads()
    bads = nd.get_bads(verbose=True)
    print('PREP analysis of artifact free data obtained from EEG_MS+RS')
    
    # Comparison with the data quality metrics
    # The proposed method:
    thresholdValues = Configuration.callThresholdValues()
    overallHighAmplitude, timeHighVarience, channelHighVarience = calculateQualtiyMeasures(data = finalEmgRaw.get_data(),thresholdValues = thresholdValues)
                                                                                            
    
    # The method:ICA with MARA
    overallHighAmplitude1, timeHighVarience1, channelHighVarience1 = calculateQualtiyMeasures(data = rawICAMARA.get_data(), thresholdValues = thresholdValues)

    method1DataQualityMetrics = {}
    method1DataQualityMetrics['OHV'] = overallHighAmplitude 
    method1DataQualityMetrics['THV'] = timeHighVarience 
    method1DataQualityMetrics['CHV'] = channelHighVarience
    
    method2DataQualityMetrics = {}
    method2DataQualityMetrics['OHV'] = overallHighAmplitude1 
    method2DataQualityMetrics['THV'] = timeHighVarience1 
    method2DataQualityMetrics['CHV'] = channelHighVarience1

    overallDataQuality = {}
    overallDataQuality['proposedMethod'] = method1DataQualityMetrics
    overallDataQuality['ICAwithMARA'] = method2DataQualityMetrics


    return overallDataQuality, rawICAMARA 
    
