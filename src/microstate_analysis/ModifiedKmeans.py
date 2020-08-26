import mpmath
import math
import argparse, os, sys, time
import numpy as np
from scipy import signal



def findGfpPeaks(data, distance=10, n_std=3):
    gfp = data.std(axis=1)
    peaks, _ = signal.find_peaks(gfp, distance=distance, height=(gfp.mean() - n_std * gfp.std(), gfp.mean() + n_std * gfp.std()))
    
    return gfp, peaks


def ModifiedKmeans(data, gfp, gfp_peaks, meanValue,  numMaps, n_runs= 10, maxerr=1e-6, maxiter=1000):
    """Modified K-means clustering as detailed in:
    [1] Pascual-Marqui et al., IEEE TBME (1995) 42(7):658--665
    [2] Murray et al., Brain Topography(2008) 20:249--264.
    Variables named as in [1], step numbering as in Table I.
    Args:
        data: numpy.array, size = number of EEG channels
        n_maps: number of microstate maps
        n_runs: number of K-means runs (optional)
        maxerr: maximum error for convergence (optional)
        maxiter: maximum number of iterations (optional)
        doplot: plot the results, default=False (optional)
    Returns:
        maps: microstate maps (number of maps x number of channels)
        L: sequence of microstate labels
        gfp_peaks: indices of local GFP maxima
        gev: global explained variance (0..1)
        cv: value of the cross-validation criterion
    """
    numTimePoints = data.shape[0]
    numChannels = data.shape[1]
    data = data - meanValue

    # GFP peaks
    gfp_values = gfp[gfp_peaks]
    gfp2 = np.sum(gfp_values**2) # normalizing constant in GEV
    n_gfp = gfp_peaks.shape[0]

    # clustering of GFP peak maps only
    V = data[gfp_peaks, :]
    sumV2 = np.sum(V**2)

    # store results for each k-means run
    cv_list =   []  # cross-validation criterion for each k-means run
    gev_list =  []  # GEV of each map for each k-means run
    gevT_list = []  # total GEV values for each k-means run
    maps_list = []  # microstate maps for each k-means run
    L_list =    []  # microstate label sequence for each k-means run
    
    for run in range(n_runs):
        # initialize random cluster centroids (indices w.r.t. n_gfp)
        rndi = np.random.permutation(n_gfp)[:n_maps]
        maps = V[rndi, :]
        # normalize row-wise (across EEG channels)
        maps /= np.sqrt(np.sum(maps**2, axis=1, keepdims=True))
        # initialize
        n_iter = 0
        var0 = 1.0
        var1 = 0.0
        # convergence criterion: variance estimate (step 6)
        while ( (np.abs((var0-var1)/var0) > maxerr) and (n_iter < maxiter) ):
            # (step 3) microstate sequence (= current cluster assignment)
            C = np.dot(V, maps.T)
            C /= (numChannels*np.outer(gfp[gfp_peaks], np.std(maps, axis=1)))
            L = np.argmax(C**2, axis=1)
            # (step 4)
            for k in range(n_maps):
                Vt = V[L==k, :]
                # (step 4a)
                Sk = np.dot(Vt.T, Vt)
                # (step 4b)
                evals, evecs = np.linalg.eig(Sk)
                v = evecs[:, np.argmax(np.abs(evals))]
                v = np.real(v)
                maps[k, :] = v/np.sqrt(np.sum(v**2))
            # (step 5)
            var1 = var0
            var0 = sumV2 - np.sum(np.sum(maps[L, :]*V, axis=1)**2)
            var0 /= (n_gfp*(numChannels-1))
            n_iter += 1
        

        # CROSS-VALIDATION criterion for this run (step 8)
        C_ = np.dot(data, maps.T)
        C_ /= (numChannels*np.outer(gfp, np.std(maps, axis=1)))
        L_ = np.argmax(C_**2, axis=1)
        var = np.sum(data**2) - np.sum(np.sum(maps[L_, :]*data, axis=1)**2)
        var /= (numTimePoints*(numChannels-1))
        cv = var * (numChannels-1)**2/(numChannels-n_maps-1.)**2

        # GEV (global explained variance) of cluster k
        gev = np.zeros(n_maps)
        for k in range(n_maps):
            r = L==k
            gev[k] = np.sum(gfp_values[r]**2 * C[r,k]**2)/gfp2
        gev_total = np.sum(gev)

        # store
        cv_list.append(cv)
        gev_list.append(gev)
        gevT_list.append(gev_total)
        maps_list.append(maps)
        L_list.append(L_)

    # select best run
    k_opt = np.argmin(cv_list)
    #k_opt = np.argmax(gevT_list)
    maps = maps_list[k_opt]
    # ms_gfp = ms_list[k_opt] # microstate sequence at GFP peaks
    gev = gev_list[k_opt]
    L_ = L_list[k_opt]

   
    return maps, L_, gfp_peaks, gev, cv
    




    









     





