import numpy as np
import ModifiedKmeans
import FindOptimalCluster as fcluster


# Function to conduct EEG microstate analysis on the  raw data for finding optimal number microstate classes or maps 
def analyzeMicrostate(): 
    subjectWiseData, optimalCluster = fcluster.findOptimalCluster()
    data = subjectWiseData.mean(axis = 0).T
    optimalMaps, labels, gfp_peaks, gev, cv = ModifiedKmeans.kmeans(data, optimalCluster, n_runs = 50, maxerr = 1e-6, 
                                                      maxiter = 1000, doplot = False)
    return data, optimalMaps, optimalCluster
    

   


   

     
   

   
    
       



    
    