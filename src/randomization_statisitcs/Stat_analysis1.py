import numpy as np


def quantifyMicrostates(maps, labels):
    # Count of total time points when each microstate class occurs
    quantifiersMicrostateClass = {}

    for i in range(len(maps)):
        quantifiersMicrostateClass['gfp'] = np.std(maps-np.mean(maps,axis=1,keepdims=True), axis = 1,
                                                   keepdims =True)
        quantifiersMicrostateClass['Count of time points of Microstate class '+str(i)] = labels.count(i)
        quantifiersMicrostateClass['Onset of Microstate class '+str(i)] = labels.index(i)
        quantifiersMicrostateClass['Offset of Microstate class '+str(i)] = max(
                                                        [j for j, e in enumerate(labels) if e == i])
        quantifiersMicrostateClass['indices BackTraceData of Microstate class '+str(i)] = [
                                                                j for j, e in enumerate(labels) if e == i] 
  
        
    

