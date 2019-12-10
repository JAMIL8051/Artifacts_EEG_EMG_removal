import numpy as np


def locmax(x):
    """Get local maxima of 1D-array

    Args:
        x: numeric sequence
    Returns:
        m: list, 1D-indices of local maxima
    """

    dx = np.diff(x) # discrete 1st derivative
    zc = np.diff(np.sign(dx)) # zero-crossings of dx
    m = 1 + np.where(zc == -2)[0] # indices of local max.
    return m





def kmeans(data, n_maps, n_runs = 10, maxerr = 1e-6, maxiter =500):
#Arguments
#data is numpy array size = no of EEG channels
#n_maps is no of microstate maps
#n_runs number of k-means runs(optional)
#maxerr: maximum error of convergence(optional)
#maxiter: maximum number of iterations (optional)
#doplot is plot results, default false(optional)
#returns:
# maps: microstate map (no. of maps x no. of channels)
#L: sequence of microstate labesls
#gfp_peaks: indices of local GFP maxima
#gev: global explained varience (0..1)
#cv: value of the cross-validation criterion
    n_time_samples = data.shape[0]
    n_channels = data.shape[1]
    data = data - data.mean(axis =1, keepdims = True)

    #Global field power peaks(GFP peaks)
    gfp = np.std(data, axis=1)
    gfp_peaks = locmax(gfp)
    gfp_values = gfp[gfp_peaks]
    gfp2 = np.sum(gfp_values ** 2)# Normalize constant in GEV
    n_gfp = gfp_peaks.shape[0]

    #clustering of GFP peak maps only
    V = data[gfp_peaks, :]
    sumV2 = np.sum(V ** 2)

#Storing results for each k-means run
    cv_list = [] # cross-validation criterion for each k-means run
    gev_list = [] # Global explained varience(GEV) of each map for each k-means run
    gevT_list = [] # total GEV values for each k-means run
    maps_list = [] # microstate maps for each k-means run
    L_list = [] # microstate label sequence for each k-means run

    for run in range(n_runs):
        #initialize random cluster centroids (indices with respect to n_gfp)
        rndi = np.random.permutation(n_gfp)[:n_maps]
        maps = V[rndi, :]
        #normalize row -wise (across EEG channels)
        maps /= np.sqrt(np.sum(maps **2, axis=1, keepdims=True))
        #initialize
        n_iter = 0
        var0 = 1.0
        var1 = 0.0
        #convergence criterion: varience estimate (step 6)
        while((np.abs((var0-var1)/var0)> maxerr) & (n_iter < maxiter) ):
            #(step 3) microstate sequence (= current cluster assigment)

            C = np.dot(V, maps.T)
            C /= (n_channels * np.outer(gfp[gfp_peaks], np.std(maps, axis =1)))
            L = np.argmax(C ** 2, axis = 1)

            #(step 4)
            for k in range(n_maps):
                 Vt = V[L==k, :]
                 #Step 4a
                 Sk = np.dot(Vt.T, Vt)
                 #step 4b
                 evals, evecs = np.linalg.eig(Sk)
                 v = evecs[:, np.argmax(np.abs(evals))]
                 maps[k,:] = v/np.sqrt(np.sum(v **2))
            #Step 5
            var1 = var0
            var0 = sumV2 - np.sum(np.sum(maps[L,:] * V, axis=1) ** 2)
            var0 /= (n_gfp * (n_channels-1))
            n_iter += 1
            if(n_iter <maxiter):
                print("\t\tK-means run {:d}/{:d} converged after {:d} iterations." .format(run+1, n_runs, n_iter))
            else:
                print("\t\tK-means run {:d}/{:d} did NOT converge after {:d} iterations.".format(run + 1, n_runs, maxiter))


            #Cross-validation criterion for this run (step 8)
            C_ = np.dot(data, maps.T)
            C_ /= (n_channels * np.outer(gfp, np.std(maps, axis=1)))
            L_ = np.argmax(C_**2, axis=1)
            var = np.sum(data**2) - np.sum(np.sum(maps[L_,:]*data, axis=1)**2)
            var /= (n_time_samples*(n_channels-1))
            cv = var * (n_channels-1) ** 2/(n_channels-n_maps-1.)**2

            #GEV of cluster k
            gev = np.zeros(n_maps)
            for k in range(n_maps):
                r =(L==k)
                gev[k] = np.sum(gfp_values[r] ** 2 * C[r,k] **2)/gfp2
            gev_total = np.sum(gev)

            #Store
            cv_list.append(cv)
            gev_list.append(gev)
            gevT_list.append(gev_total)
            maps_list.append(maps)
            L_list.append(L_)

        #Select best run
            k_opt = np.argmin(cv_list)
            #k_opt = np.argmax(gevT_list)
            maps = maps_list[k_opt]
            #ms_gfp = ms_list[k_opt] # microstate sequence at GFP peaks
            gev = gev_list[k_opt]
            L_ = L_list[k_opt]
    return maps, L_, gfp_peaks, gev,cv





