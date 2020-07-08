import numpy as np


#Quantifiers of microstate maps from two groups/conditions

def quantifyMicrostates(maps, labels):
    # Count of total time points when each microstate class occurs
    quantifiersMicrostateClass = {}
    # 3 Parameters 
    countOfTimePoints = {}
    microstateOnset = {}
    microstateOffset = {}
    microstateIndicesBackTrace = {}

    quantifiersMicrostateClass['gfp'] = np.std(maps-np.mean(maps,axis=1,keepdims=True), axis = 1,
                                                   keepdims =True)
    
    for i in range(len(maps)):
        countOfTimePoints['class: '+str(i)] = list(labels).count(i)
        microstateOnset['class: '+str(i)] = list(labels).index(i)
        microstateOffset['class: '+str(i)] = max([j for j, e in enumerate(list(labels)) if e == i])
        microstateIndicesBackTrace['class: '+str(i)] = [j for j, e in enumerate(list(labels)) if e == i] 
    
    quantifiersMicrostateClass['Count of time points'] = countOfTimePoints  
    quantifiersMicrostateClass['Onset of Microstate classes'] = microstateOnset
    quantifiersMicrostateClass['Offset of Microstate classes'] = microstateOffset                                       
    quantifiersMicrostateClass['indices BackTraceData of Microstate classes'] = microstateIndicesBackTrace 
                                                               
    return quantifiersMicrostateClass
  


       
    

