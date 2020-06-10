import numpy as np

#Quantifiers of microstate maps from two groups/conditions
#Dot product of two vectors
def dotproduct(v1,v2):
    return sum((a*b) for a,b in zip(v1,v2))


#To get the length(magnitude) of a vector   
def length(v):
    return math.sqrt(dotproduct(v,v))


#orthogonal project of each maps of a group/condition 
def orthogonal_projection_3d(data):
    for j in range(0,data.shape[0]):
        nd = data.shape[1]
        while (nd != 3):
            nd = nd-1
            last_element = data[j][nd] # Setting the last element to 1
            t = 1/last_element
            data_scaled = np.empty((data.shape[0],nd),dtype = float, order ='F')
            for i in range(0, nd):
                data_scaled[j][i] = data[j][i]/t
    return data_scaled


#Orthogonal projection in 3d for each map
#def orthogonal_projection_3d_map(data):
#    nd = len(data)
    
#    while (nd != 3):
#        nd = nd-1
#        last_element = data[nd] # Setting the last element to 1
#        t = 1/last_element
#        data_scaled = np.empty((nd,1),dtype = float, order ='F')
#        for i in range(0, nd):
#            data_scaled[i] = data[i]/t
        
#    return data_scaled


# Topographic dissimilarity
def topo_dissimilarity(u,v):
    norm_u = normalized_vector(u)
    norm_v = normalized_vector(v)
    dissimilarity = math.sqrt(sum(((a-b)**2) for a,b in zip(norm_u, norm_v)))
    return dissimilarity


# Topographic correlation
def topographic_correlation(v1,v2):
    corr = (dotproduct(v1,v2))/(length(v1)*length(v2))
    return corr


#Function to get unit vector
def normalized_vector(u):
    norm_u =u/length(u)
    return norm_u


def quantifyMicrostates(maps, labels):
    # Count of total time points when each microstate class occurs
    quantifiersMicrostateClass = {}

    quantifiersMicrostateClass['gfp'] = np.std(maps-np.mean(maps,axis=1,keepdims=True), axis = 1,
                                                   keepdims =True)

    for i in range(len(maps)):
        
        quantifiersMicrostateClass['Count of time points of Microstate class '+str(i)] = labels.count(i)
        quantifiersMicrostateClass['Onset of Microstate class '+str(i)] = labels.index(i)
        quantifiersMicrostateClass['Offset of Microstate class '+str(i)] = max(
                                                        [j for j, e in enumerate(labels) if e == i])
        quantifiersMicrostateClass['indices BackTraceData of Microstate class '+str(i)] = [
                                                                j for j, e in enumerate(labels) if e == i] 
  
        
    

