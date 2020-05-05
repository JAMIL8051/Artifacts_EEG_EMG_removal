import mpmath
import math
import argparse, os, sys, time
import numpy as np
import rcca
import matplotlib.pyplot as plt
#from palettable.colorbrewer import qualitative
from scipy.interpolate import griddata
from scipy import signal
from scipy.signal import butter, filtfilt, welch
import scipy.stats
import pandas as pd
#import statsmodels.api as sm
#from statsmodels.formula.api import ols
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA

#Edf file reader function
def read_edf(filename):
    """Basic EDF file format reader

    EDF specifications: http://www.edfplus.info/specs/edf.html

    Args:
        filename: full path to the '.edf' file
    Returns:
        chs: list of channel names
        fs: sampling frequency in [Hz]
        data: EEG data as numpy.array (samples x channels)
    """

    def readn(n):
        """read n bytes."""
        return np.fromfile(fp, sep='', dtype=np.int8, count=n)

    def bytestr(bytes, i):
        """convert byte array to string."""
        return np.array([bytes[k] for k in range(i*8, (i+1)*8)]).tostring()

    fp = open(filename, 'r')
    x = np.fromfile(fp, sep='', dtype=np.uint8, count=256).tostring()
    header = {}
    header['version'] = x[0:8]
    header['patientID'] = x[8:88]
    header['recordingID'] = x[88:168]
    header['startdate'] = x[168:176]
    header['starttime'] = x[176:184]
    header['length'] = int(x[184:192]) # header length (bytes)
    header['reserved'] = x[192:236]
    header['records'] = int(x[236:244]) # number of records
    header['duration'] = float(x[244:252]) # duration of each record [sec]
    header['channels'] = int(x[252:256]) # ns - number of signals
    n_ch = header['channels']  # number of EEG channels
    header['channelname'] = (readn(16*n_ch)).tostring()
    header['transducer'] = (readn(80*n_ch)).tostring().split()
    header['physdime'] = (readn(8*n_ch)).tostring().split()
    header['physmin'] = []
    b = readn(8*n_ch)
    for i in range(n_ch): header['physmin'].append(float(bytestr(b, i)))
    header['physmax'] = []
    b = readn(8*n_ch)
    for i in range(n_ch): header['physmax'].append(float(bytestr(b, i)))
    header['digimin'] = []
    b = readn(8*n_ch)
    for i in range(n_ch): header['digimin'].append(int(bytestr(b, i)))
    header['digimax'] = []
    b = readn(8*n_ch)
    for i in range(n_ch): header['digimax'].append(int(bytestr(b, i)))
    header['prefilt'] = (readn(80*n_ch)).tostring().split()
    header['samples_per_record'] = []
    b = readn(8*n_ch)
    for i in range(n_ch): header['samples_per_record'].append(float(bytestr(b, i)))
    nr = header['records']
    n_per_rec = int(header['samples_per_record'][0])
    n_total = int(nr*n_per_rec*n_ch)
    fp.seek(header['length'],os.SEEK_SET)  # header end = data start
    data = np.fromfile(fp, sep='', dtype=np.int16, count=n_total)  # count=-1
    fp.close()

    # re-order
    data = np.reshape(data,(n_per_rec,n_ch,nr),order='F')
    data = np.transpose(data,(0,2,1))
    data = np.reshape(data,(n_per_rec*nr,n_ch),order='F')

    # convert to physical dimensions
    for k in range(data.shape[1]):
        d_min = float(header['digimin'][k])
        d_max = float(header['digimax'][k])
        p_min = float(header['physmin'][k])
        p_max = float(header['physmax'][k])
        if ((d_max-d_min) > 0):
            data[:,k] = p_min+(data[:,k]-d_min)/(d_max-d_min)*(p_max-p_min)

    print(header)
    return header['channelname'].split(),\
           header['samples_per_record'][0]/header['duration'],\
           data


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


def findstr(s, L):
#    """Find string in list of strings, returns indices.
#
#    Args:
#        s: query string
#        L: list of strings to search
#    Returns:
#        x: list of indices where s is found in L
#    """
#
    x = [i for i, l in enumerate(L) if (l==s)]
    return x


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

#Function for calculating band power
def bandpower(data, sf, band, window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simps
    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp





def bp_filter(data, f_lo,f_hi, fs):
    """Digital 6th order butterworth band pass filter
     Args:
        data: numpy.array, time along axis 0
        (f_lo, f_hi): frequency band to extract [Hz]
        fs: sampling frequency [Hz]
    Returns:
        data_filt: band-pass filtered data, same shape as data
    """
    data_filt = np.zeros_like(data)
    f_ny = fs/2.  # Nyquist frequency
    b_lo = f_lo/f_ny  # normalized frequency [0..1]
    b_hi = f_hi/f_ny  # normalized frequency [0..1]
    p_lp = {"N":6, "Wn":b_hi, "btype":"lowpass", "analog":False, "output":"ba"}
    p_hp = {"N":6, "Wn":b_lo, "btype":"highpass", "analog":False, "output":"ba"}
    bp_b1, bp_a1 = butter(**p_lp)
    bp_b2, bp_a2 = butter(**p_hp)
    data_filt = filtfilt(bp_b1, bp_a1, data, axis=0)
    data_filt = filtfilt(bp_b2, bp_a2, data_filt, axis=0)
    return data_filt


#Excluding the zero_mean
def exclude_zero_mean(data):
    sum = np.sum(data, axis = 0)
    col = (data!=0).sum(0)
    reject_zero_mean = sum/col
    return reject_zero_mean


def topo(data, n_grid=64):
    """Interpolate EEG topography onto a regularly spaced grid

    Args:
        data: numpy.array, size = number of EEG channels
        n_grid: integer, interpolate to n_grid x n_grid array, default=64
    Returns:
        data_interpol: cubic interpolation of EEG topography, n_grid x n_grid
                       contains nan values
    """
    channels, locs = read_xyz('cap.xyz')
    n_channels = len(channels)
    #locs /= np.sqrt(np.sum(locs**2,axis=1))[:,np.newaxis]
    locs /= np.linalg.norm(locs, 2, axis=1, keepdims=True)
    c = findstr('Cz', channels)[0]
    # print 'center electrode for interpolation: ' + channels[c]
    #w = np.sqrt(np.sum((locs-locs[c])**2, axis=1))
    w = np.linalg.norm(locs - locs[c], 2, axis=1)
    #arclen = 2*np.arcsin(w/2)
    arclen = np.arcsin(w / 2. * np.sqrt(4. - w * w))
    it1 = locs[:,0] - locs[c][0]
    it2 = locs[:,1] - locs[c][1]
    mapped = map(complex, it1, it2)
    listMapped = list(mapped)
    phi = np.angle(listMapped)
    X = arclen * np.real(np.exp(1j * phi))
    Y = arclen * np.imag(np.exp(1j * phi))
    r = max([max(X),max(Y)])
    Xi = np.linspace(-r,r,n_grid)
    Yi = np.linspace(-r,r,n_grid)
    data_ip = griddata((X, Y), data, (Xi[None,:], Yi[:,None]), method='cubic')
    return data_ip


def eeg2map(data):
    """Interpolate and normalize EEG topography, ignoring not a number(nan) values

    Args:
        data: numpy.array, size = number of EEG channels
        n_grid: interger, interpolate to n_grid x n_grid array, default=64
    Returns:
        top_norm: normalized topography, n_grid x n_grid
    """
    n_grid = 64
    top = topo(data, n_grid)
    mn = np.nanmin(top)
    mx = np.nanmax(top)
    top_norm = (top-mn)/(mx-mn)
    return top_norm


def kmeans(data, n_maps, n_runs= 10, maxerr=1e-6, maxiter=1000, doplot = False):
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
    n_t = data.shape[0]
    n_ch = data.shape[1]
    data = data - data.mean(axis=1, keepdims = True)

    # GFP peaks
    gfp = np.std(data, axis=1)
    gfp_peaks = locmax(gfp)
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
        while ( (np.abs((var0-var1)/var0) > maxerr) & (n_iter < maxiter) ):
            # (step 3) microstate sequence (= current cluster assignment)
            C = np.dot(V, maps.T)
            C /= (n_ch*np.outer(gfp[gfp_peaks], np.std(maps, axis=1)))
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
            var0 /= (n_gfp*(n_ch-1))
            n_iter += 1
        if (n_iter < maxiter):
            print("\t\tK-means run {:d}/{:d} converged after {:d} iterations.".format(run+1, n_runs, n_iter))
        else:
            print("\t\tK-means run {:d}/{:d} did NOT converge after {:d} iterations.".format(run+1, n_runs, maxiter))

        # CROSS-VALIDATION criterion for this run (step 8)
        C_ = np.dot(data, maps.T)
        C_ /= (n_ch*np.outer(gfp, np.std(maps, axis=1)))
        L_ = np.argmax(C_**2, axis=1)
        var = np.sum(data**2) - np.sum(np.sum(maps[L_, :]*data, axis=1)**2)
        var /= (n_t*(n_ch-1))
        cv = var * (n_ch-1)**2/(n_ch-n_maps-1.)**2

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

    if doplot:
        plt.ion()
        # matplotlib's perceptually uniform sequential colormaps:
        # magma, inferno, plasma, viridis
        cm = plt.cm.magma
        fig, axarr = plt.subplots(1, n_maps, figsize=(20,5))
        fig.patch.set_facecolor('white')
        for imap in range(n_maps):
            axarr[imap].imshow(eeg2map(maps[imap, :]), cmap=cm, origin='lower')
            axarr[imap].set_xticks([])
            axarr[imap].set_xticklabels([])
            axarr[imap].set_yticks([])
            axarr[imap].set_yticklabels([])
        title = "K-means cluster centroids"
        axarr[0].set_title(title, fontsize=16, fontweight="bold")
        plt.show()

        # --- assign map labels manually ---
        order_str = raw_input("\n\t\tAssign map labels (e.g. 0, 2, 1, 3): ")
        order_str = order_str.replace(",", "")
        order_str = order_str.replace(" ", "")
        if (len(order_str) != n_maps):
            if (len(order_str)==0):
                print("\t\tEmpty input string.")
            else:
                print("\t\tParsed manual input: {:s}".format(", ".join(order_str)))
                print("\t\tNumber of labels does not equal number of clusters.")
            print("\t\tContinue using the original assignment...\n")
        else:
            order = np.zeros(n_maps, dtype=int)
            for i, s in enumerate(order_str):
                order[i] = int(s)
            print("\t\tRe-ordered labels: {:s}".format(", ".join(order_str)))
            # re-order return variables
            maps = maps[order,:]
            for i in range(len(L)):
                L[i] = order[L[i]]
            gev = gev[order]
            # Figure
            fig, axarr = plt.subplots(1, n_maps, figsize=(20,5))
            fig.patch.set_facecolor('white')
            for imap in range(n_maps):
                axarr[imap].imshow(eeg2map(maps[imap, :]), cmap=cm, origin='lower')
                axarr[imap].set_xticks([])
                axarr[imap].set_xticklabels([])
                axarr[imap].set_yticks([])
                axarr[imap].set_yticklabels([])
            title = "re-ordered K-means cluster centroids"
            axarr[0].set_title(title, fontsize=16, fontweight="bold")
            plt.show()
            plt.ioff()
    return maps, L_, gfp_peaks, gev, cv
    

#Dot product of two vectors
def dotproduct(v1,v2):
    return sum((a*b) for a,b in zip(v1,v2))


#To get the length(magnitude) of a vector   
def length(v):
    return math.sqrt(dotproduct(v,v))


#orthogonal project of each maps of a group
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
def orthogonal_projection_3d_map(data):
    nd = len(data)
    
    while (nd != 3):
        nd = nd-1
        last_element = data[nd] # Setting the last element to 1
        t = 1/last_element
        data_scaled = np.empty((nd,1),dtype = float, order ='F')
        for i in range(0, nd):
            data_scaled[i] = data[i]/t
        
    return data_scaled


def topo_dissimilarity(u,v):
    norm_u = normalized_vector(u)
    norm_v = normalized_vector(v)
    dissimilarity = math.sqrt(sum(((a-b)**2) for a,b in zip(norm_u, norm_v)))
    return dissimilarity


def topographic_correlation(v1,v2):
    corr = (dotproduct(v1,v2))/(length(v1)*length(v2))
    return corr


#Spatial analysis function
def spatial_derivative(data_bad, data_grp1):
    data = data_bad
    x = np.linspace(1,len(data), num = len(data))
    data2 = np.empty((len(data),1),dtype = float, order ='F')
    for i in range(len(data)-1):
        data2[i] =data[i+1]-data[i]
    
    data3 = data2
    data4 = np.empty((len(data),1),dtype = float, order='F')
    for i in range(len(data)-1):
        data4[i] = data3[i+1]-data[i]
    
    data_grp1_2 = np.empty((len(data),1),dtype = float, order ='F')
    for i in range(len(data)-1):
        data2[i] =data_grp1[i+1]-data_grp1[i]
    
    data_grp1_3 = data_grp1_2
    
    data_grp1_4 = np.empty((len(data),1),dtype = float, order='F')
    for i in range(len(data)-1):
        data4[i] = data_grp1_3[i+1]-data_grp1_3[i]
    
    return x,data2, data4, data_grp1_2,data_grp1_4


#Function to get unit vector
def normalized_vector(u):
    norm_u =u/length(u)
    return norm_u


def test_topography_consistancy(data1, data2):
    #This function test the topographic consistency among the maps of the two groups:accross the two groups
    
    data = np.concatenate((data1,data2), axis = 0)# Here axis=0/1 will depend whether we shuffle potential 
    #values of same electrode/channel or just reshuffle electordes. If potential values then axis=0 else 1 
    #for reshuffle of electrodes/channels.
    
    observed_grand_mean = np.mean(data, axis = 0)
    gfp_observed_grand_mean = np.std(observed_grand_mean)
    gfp_after_shuffle = np.zeros(5000)

    for i in range(5000):
        np.random.shuffle(data1.T)# Here data1/data1.T will depend whether we shuffle potential values of 
        #same electrode/channel or just reshuffle electordes. 
        #If potential values then data1 else data1.T for reshuffle of electrodes/channels.
        np.random.shuffle(data2.T)
        
        grand_mean_after_shuffle = np.mean(np.concatenate((data1.T,data2.T),axis = 0))
        gfp_after_shuffle[i] = np.std(grand_mean_after_shuffle)
        
    count = 0

    for i in range(len(gfp_after_shuffle)):
        if gfp_after_shuffle[i] >= gfp_observed_grand_mean:
            count = count+1

    percentage = (count/5000)
    print('\nThe probability of null hypothesis for topographic consistency is :{}'.format(percentage))
    return gfp_after_shuffle


def comparison_map_diff_between_two_conditions(data1,data2):
    mean_data1 = np.mean(data1,axis = 0,keepdims = True)
    mean_data2 = np.mean(data2,axis = 0,keepdims = True)
    diff_map = mean_data1 - mean_data2
    #norm_mean_data1 = normalized_vector(mean_data1)
    #norm_mean_data2 = normalized_vector(mean_data1)
    #norm_diff_map = norm_mean_data1 - norm_mean_data2
    gfp_diff_map = np.std(diff_map)
    #gmd = np.std(norm_diff_map) # Global map dissimilarity
    rand_gfp_diff_map =np.zeros(5000)
    #Concatenate data
    data = np.concatenate((data1, data2), axis = 0)
    for i in range(5000):
        #dataA = np.random.permutation(data1.T)
        #dataB = np.random.permutation(data2.T)

        np.random.shuffle(data)
        dataA  = data[0:data1.shape[0], 0:data1.shape[1]]
        dataB  = data[data1.shape[0]:2*data1.shape[0], 0:data1.shape[1]]

        #lensA = np.array(map(len,dataA)) # Thanks to @Kasramvd on this!
        #valsA = np.concatenate(dataA)
        #shift_idxA = np.append(0,lensA[:-1].cumsum())
        #mean_dataA = np.add.reduceat(valsA,shift_idxA)/lensA.astype(float)
        
        #lensB = np.array(map(len,dataB)) # Thanks to @Kasramvd on this!
        #valsB = np.concatenate(dataB)
        #shift_idxB = np.append(0,lensB[:-1].cumsum())
        #mean_dataB = np.add.reduceat(valsB,shift_idxB)/lensB.astype(float)
        
        
        mean_dataA = np.mean(dataA, axis = 0, keepdims = True)
        mean_dataB = np.mean(dataB, axis = 0, keepdims = True)
        rand_diff_map = mean_dataA - mean_dataB
        rand_gfp_diff_map[i] = np.std(rand_diff_map)
    #print(rand_gfp_diff_map)
    
    count_map_diff_two = 0
    for i in range(0,len(rand_gfp_diff_map)):
        if rand_gfp_diff_map[i] >= gfp_diff_map:
            count_map_diff_two = count_map_diff_two + 1

    probability = count_map_diff_two/len(rand_gfp_diff_map)
    print(gfp_diff_map)
    print('\nThe probability that the "difference between two conditions/groups are observed by: chance" or the probability of NULL hyothesis is: {}'.format(probability))
    
    return rand_gfp_diff_map


#Formatting the data into conditions and subjects
def format_data(data, condition, n_subject, n_ch, ith_class):
        n_condition = len(condition)
        data_temp = np.zeros((n_subject, n_condition, n_ch))
        for i in range(n_subject):
            for j in range(n_condition):
                data_temp[i,j] = data[condition[j]][i][ith_class]
        return data_temp


#Shuffling the indices of the data matrix for easy and faster shuffle of the data matrix
def shuffle_data(data, n_condition):
    for i in range(data.shape[0]):
        random_index = np.random.permutation(n_condition)
        data[i] = data[i][random_index]
    return data


def comparison_map_diff_across_conditions(data, condition, n_subject, n_ch, ith_class):
    data_temp = format_data(data, condition, n_subject, n_ch, ith_class) 
    data = data_temp    
    grand_mean_across_subjects = np.mean(data, axis=0)
    grand_mean_across_subjects_across_conditions = np.mean(data, axis=(0, 1))

    residual = np.power(grand_mean_across_subjects - grand_mean_across_subjects_across_conditions, 2)
    generalized_dissimilarity = np.sum(np.sqrt(np.mean(residual, axis=1)))
    
    residual_maps = grand_mean_across_subjects - grand_mean_across_subjects_across_conditions
    #Within-subject factor:Shuffle accross subjects:
    rand_effect_size = np.zeros(5000)
    for i in range(5000):
        data_s = shuffle_data(residual_maps, n_condition = 6)
        rand_grand_mean_across_subjects = np.mean(data_s, axis=0)
        rand_grand_mean_across_subjects_across_conditions = np.mean(data_s, axis=(0, 1))
        residual = np.power(grand_mean_across_subjects - grand_mean_across_subjects_across_conditions, 2)
        rand_generalized_dissimilarity = np.sum(np.sqrt(np.mean(residual, axis=1)))
        rand_effect_size[i] = rand_generalized_dissimilarity

    count_map_diff_multi = 0

    for i in range(0,len(rand_effect_size)):
        if rand_effect_size[i] >= observed_effect_size:
            count_map_diff_multi = count_map_diff_multi + 1

    percentage = count_map_diff_multi/5000
    print('\nThe probability of null hypothesis across different conditions is: {}'.format(percentage))
    return rand_effect_size


def effects_conditions(data1,data2,data3):
#This function shows: a) The effect for selection of bad channels as group 
#b) Effect for group1 and group 2 channels to see whether these are of bad channels group 
#c) Effect of interaction of these two factors:Bad channels group and group1,group2 channels
    
    data = np.concatenate((data1,data2,data3), axis = 0)
    grand_mean = np.mean(data,axis = 0)

    #Step 2 page 178 Chap 8 Electrical Neuroimaging book 
    res_maps1 = data1 - grand_mean
    res_maps2 = data2 - grand_mean
    res_maps3 = data3 - grand_mean
     
    #Step 3a.
    grand_mean_res_map1 = np.mean(res_maps1, axis = 0)
    grand_mean_res_map2 = np.mean(res_maps2, axis = 0)
    grand_mean_res_map3 = np.mean(res_maps3, axis = 0)

    #Step 3b.
    obs_effect_size =np.std(grand_mean_res_map1)+ np.std(grand_mean_res_map3)+ np.std(grand_mean_res_map3)
    
    #Step 3c. page 178 Chap 8 Electrical Neuroimaging book
    res_data1 = data1- grand_mean_res_map1
    res_data2 = data2- grand_mean_res_map2
    res_data3 = data3- grand_mean_res_map3

    rand_effect_size = np.zeros(5000)
    rand_effect_size1 = np.zeros(5000)
    rand_effect_size2 = np.zeros(5000)
    
    rand_res_data = np.concatenate((res_maps1,res_maps2,res_maps3),axis = 0)
    for i in range(5000):
        np.random.shuffle(rand_res_data)
        rand_grand_mean_res_map1 = np.mean(rand_res_data[0:data1.shape[0],:], axis = 0)
        rand_grand_mean_res_map2 = np.mean(rand_res_data[data1.shape[0]:2*data1.shape[0],:], axis = 0)
        rand_grand_mean_res_map3 = np.mean(rand_res_data[2*data1.shape[0]:3*data1.shape[0],:], axis = 0)
        
        rand_effect_size[i] = np.std(rand_grand_mean_res_map1)
        rand_effect_size1[i] = np.std(rand_grand_mean_res_map3)
        rand_effect_size2[i] = np.std(rand_grand_mean_res_map3) 
        #Step 3 repeat
        rand_res_data1 = data1- rand_grand_mean_res_map1
        rand_res_data2 = data2- rand_grand_mean_res_map2
        rand_res_data3 = data3- rand_grand_mean_res_map3

    count_con1 = 0
    for i in range(0,len(rand_effect_size)):
        if rand_effect_size[i]>=obs_effect_size:
            count_con1 = count_con1+1
    percentage = count_con1/5000
    print('\nThe probability of the null hypothesis for bad channels group factor:{}'.format(count_con1))
    
    count_con2 = 0
    for i in range(0,len(rand_effect_size)):
        if rand_effect_size1[i]>=obs_effect_size:
            count_con2 = count_con2 +1
    percentage1 = count_con2/5000
    print('\nThe probability of the null hypothesis for channels group 1 factor:{}'.format(count_con2))
    
    count_con3 = 0
    for i in range(0,len(rand_effect_size)):
        if rand_effect_size2[i]>=obs_effect_size:
            count_con3  = count_con3 +1
    percentage2 = count_con3/5000
    print('\nThe probability of the null hypothesis for channels group 2 factor:{}'.format(count_con3))

    return percentage, percentage1, percentage2


def freq_effects_test(data1,data2,data3,p=1):
    #this is for the how likely we get the significant results by chance. 
    #Input agrs: data1/data2/dat3: vector(array) of shape no.of randomizations by 1 
    # p = the p-value percentage of threshold. By default 1% 
    first_run1 = data1[0]
    first_run2 = data2[0]
    first_run3 = data3[0]
    
    pseudo_sig1 = 0
    pseudo_sig2 = 0
    pseudo_sig3 = 0

    if p == 5:
        data1 = data1[0:1000]
        data2 = data2[0:1000]
        data3 = data3[0:1000]

    for i in range(1,len(data1)):
        if data1[i]>=first_run1:
            pseudo_sig1=pseudo_sig1+1
    percentage_freq1 = pseudo_sig1/len(data1)
    plt.hist2d(pseudo_sig1, bins = 10)
    plt.show()

    for i in range(1,len(data2)):
        if data2[i]>=first_run2:
            pseudo_sig1=pseudo_sig1+1
    percentage_freq2 = pseudo_sig1/len(data2)
    plt.hist(pseudo_sig2, bins = 10)
    plt.show()

    for i in range(1,len(data3)):
        if data1[i]>=first_run1:
            pseudo_sig1=pseudo_sig1+1
    percentage_freq3 = pseudo_sig1/len(data3)
    plt.hist(pseudo_sig3, bins = 10)
    plt.show()

    return percentage_freq1, percentage_freq2, percentage_freq3


#Function for spataial correlation
def spatial_correlation(averaged_v1, averaged_v2, std_v1, std_v2, n_ch):
    correlation = np.dot(averaged_v1, averaged_v2) / (n_ch * np.outer(std_v1, std_v2))
    return correlation


#Function to back fit the microstate maps on the raw data
def fit_back(data, maps, distance= 10, n_std=3, polarity=False):
    gfp = data.std(axis=1)
    peaks, _ = signal.find_peaks(gfp, distance=distance, height=(gfp.mean() - n_std * gfp.std(), gfp.mean() + n_std * gfp.std()))
    label = np.full(data.shape[0], -1)
    correlation = spatial_correlation(data[peaks], zero_mean(maps, 1).T, gfp[peaks], maps.std(axis=1), data.shape[0])
    correlation = correlation if polarity else abs(correlation)
    label_peaks = np.argmax(correlation, axis=1)
    for i in range(len(peaks)):
        if i == 0:
            previous_middle = 0
            next_middle = int((peaks[i] + peaks[i + 1]) / 2)
        elif i == len(peaks) - 1:
            previous_middle = int((peaks[i] + peaks[i - 1]) / 2)
            next_middle = len(peaks) - 1
        else:
            previous_middle = int((peaks[i] + peaks[i - 1]) / 2)
            next_middle = int((peaks[i] + peaks[i + 1]) / 2)
        label[previous_middle:next_middle] = label_peaks[i]
    return label


#Formation of Data matrix for Partial Least Square
def concatenate_data_by_condition(data, task_condition, exclude_task):
    res = OrderedDict()
    for condition in task_condition:
        for task_name, task_data in data.items():
            if task_name not in exclude_task:
                if condition == task_name.split("_")[1]:
                    res[condition] = np.asarray(task_data) if condition not in res else np.concatenate((res[condition], np.asarray(task_data)), axis=1)
    return res


def pls(data, task_condition, exclude_task, condition, n_subject,n_ch):
    """ Spatio-temporal Partial least square: As detailed in the reference. 
        [1]Partial least squares analysis of neuroimaging data: applications and advances.
        By Anthony Randal McIntosha,* and Nancy J. Lobaughb 
        aRotman Research Institute of Baycrest Centre, University of Toronto, Toronto, Ontario, Canada M6A 2E1
        bSunnybrook and Women’s College Health Sciences Centre, 
        University of Toronto, Toronto, Ontario, Canada M4N 3M5

        Args: 
            data: numpy.array, size = no. of time samples (time_samples) by number of EEG channels (n_ch)
        
    """ 
    res = concatenate_data_by_condition(data, task_condition, exclude_task)
    #No. of time samples. data parameter should have all the time samples as rows and n_ch as columns 
 
    time_samples = data.shape[0]#Ideally the data parameter should contain the no. of time samples
    n_condition = len(condition)
    data_temp = np.zeros((n_condition, n_subject, t*n_ch))
    for i in range(n_condition):
        for j in range(n_subject):
            data_temp[i,j] = res[condition].flatten('F')
    

    #Design_matrix:Contrast matrix
    design_mat = np.zeros((n_subject,n_condition,n_condition-1),dtype ='int') 
    for subject in range(n_subject):
        val = np.arange(n_condition,0,-1)
        val = np.diag(val)
        temp = (1)*np.ones((n_condition,n_condition),dtype='int')
        val = val + temp
        val = val[:,0:n_condition-1]
        contrast_mat = np.tril(val, k=0)
        design_mat[i,:,:] = contrast_mat

    design_mat = design_mat.reshape((n_subject*n_condition,n_condition-1))

    #Data matrix:It should contain n*k rows where n=no.of subject and k = no.of conditions to compare
    #and each row of the matrix will contain t*n_ch columns 
    #where t=no.of time samples and n_ch=no.of channels
     
    M = data_temp.reshape((n_subject*n_condition,time_samples*n_ch))
    vec_ones = np.ones((n_subjects*n_condition),dtype ='int')
    I = vec_ones
    C_trans = design_mat.T 
    part = M - np.dot(I,(np.dot(I.T,M)/n_subject*n_condition))
    part = part/(n_condition-1)
    Y = np.dot(C_trans, part)

#Application of Singular value decomposition(SVD)
    u,s,vh = np.linalg.svd(Y.T,full_matrices = False)
    obs_effect_size = s
    rand_effect_size = np.zeros((5000,len(s)),dtype = 'float')
    for i in range(5000):
        rand_des_mat = shuffle_data(design_mat, n_condition)
        rand_C_trans = rand_des_mat.T
        rand_part = M - np.dot(I,(np.dot(I.T,M)/n_subject*n_condition))
        rand_part = rand_part/(n_condition-1)
        rand_Y = np.dot(rand_C_trans,rand_part)
        rand_u,rand_s,rand_vh = np.linalg.svd(rand_Y.T,full_matrices = False, compute_uv = True)
        rand_effect_size[i] = rand_s

    count_LV1 = 0
    count_LV2 = 0
    rand_effect_size_mod1 = rand_effect_size[:,0]
    rand_effect_size_mod2 = rand_effect_size[:,1]
    for i in range(0,len(rand_effect_size)):
        if(rand_effect_size_mod1[i]>=obs_effect_size[0]):
            count_LV1 += 1
        if(rand_effect_size_mod2[i]>=obs_effect_size[0]):
            count_LV2 += 1

    p_value_LV1 = count_LV1/5000
    p_value_LV2 = count_LV2/5000

    return p_value_LV1, p_value_LV2

     
def oneway_anova(data1,data2,data3):
    
    for i in range(0,len(data1)):
        Class_A = list(data1[i])+ list(data2[i])+list(data3[i])
        group_names = (['Bad']*len(data1[i])+(['Group1']*len(data2[i]))+(['Group2']*len(data3[i])))
        data = pd.DataFrame({'Group_names':group_names, 'Microstate_class_A': Class_A})
        data.groupby('Group_names').mean()
        
        lm = ols('Microstate_class_A ~ C(Group_names)', data = data).fit()
        print(lm.summary())
        table = sm.stats.anova_lm(lm)
        print('\t\t 1-way ANOVA table')
        print(table)

        #Computing the overall mean
        overall_mean = data['Microstate_class_A'].mean()
        print(overall_mean)
        #Computing the Sum of Squares Total
        data['overall_mean'] = overall_mean
        ss_total = sum((data['Microstate_class_A']-data['overall_mean'])**2)
        print(ss_total)

        #Computing thre group means
        group_means = data.groupby('Group_names').mean()
        group_means = group_means.rename(columns = {'Microstate_class_A': 'group_mean'})
        print(group_means)

        #Addition of group means and overall means to the original data frame
        data = data.merge(group_means, left_on = 'Group_names', right_index= True)

        #Computing Sum of Squares Residual
        ss_residual = sum((data['Microstate_class_A']-data['group_mean'])**2)
        print(ss_residual)

        #Computation of sum of sqaures model
        #ss_explained = sum((data['overall_mean']-data['group_mean'])**2)
        ss_explained = ss_total - ss_residual
        print(ss_explained)

        #Computation of mean sqaure residual
        n_groups = len(set(data['Group_names']))
        n_obs = data.shape[0]
        df_residual = n_obs - n_groups
        ms_residual = ss_residual/df_residual
        print(ms_residual)

        #Computation of mean square exaplained
        df_explained = n_groups -1
        ms_explained = ss_explained/df_explained
        print(ms_explained)

        #Computation of F-value
        f = ms_explained/ms_residual
        print(f)

        #Computing the p-value
        p_value = 1-scipy.stats.f.cdf(f, df_explained,df_residual)
        print(p_value)

    return None


def pca_app(data):
    pca = PCA(n_components = 3)
    pca.fit(data)
    print("Result of explained varience ratio")
    print(pca.explained_variance_ratio_)
    print("Singular values after PCA")
    print(pca.singular_values_)

    return None


def p_empirical(data, n_clusters):
    "Empirical symbol distribution"
    #Arg: data: numpy array with size of length of microstate sequence
    # n_clusters: no. of ms clusters
    #returns: p: empirical distribution

    p = np.zeros(n_clusters)
    n = len(data)
    for i in range(n):
        p[data[i]] += 1.0
    p /= n
    return p


def T_empirical(data, n_clusters):
    """Empirical transition of the maps from one class to another.
    
    Args: data: numpy array of the length of ms sequence obtained
          n_clusters: no. of ms maps or class or clusters
    
    Returns: T: empirical transition matrix
        """
    T = np.zeros((n_clusters,n_clusters))
    n = len(data)
    for i in range(n-1):
        T[data[i],data[i+1]] += 1.0
    p_row = np.sum(T, axis = 1)
    for i in range(n_clusters):
        if (p_row[i] != 0.0):
            for j in range(n_clusters):
                T[i,j]/=p_row[i]
    return T


def print_matrix(T):
    """Console output of T matrix
    Args: T: matrix to print
    """
    for i,j in np.ndindex(T.shape):
        if(j==0):
            print("\t\t[{:.3f}".format(T[i,j]))
        elif (j == T.shape[1]-1):
            print("{:.3f}]\n".format(T[i,j]))
        else:
            print("{:.3f}".format(T[i,j]))
    return None


def rcanonical(data1,data2):
    nSamples = len(data1)
    train1 = data1[:int(nSamples//2)]
    train2 = data2[:int(nSamples//2)]
    
    test1 = data1[int(nSamples//2):]
    test2 = data2[int(nSamples//2):]
    n_components = 3
    cca = rcca.CCA(kernelcca = False, reg = 0., numCC = n_components)
 
    cca.train([train1,train2])
    testcorrs = cca.validate([test1,test2])

    plt.plot(np.arrange(n_components)+1, cca.cancorrs, 'ko')
    plt.xlim(0.5, 0.5+n_components)
    plt.xticks(np.arange(n_components)+1)
    plt.xlabel('Canonical component')
    plt.ylabel('Canonical correlattions')
    plt.title('Canonical correlations')
    print('''The canonical correlations are:\n
    Component 1: %0.02f\n
    Component 2: %0.02f\n
    Component 3: %0.02f\n
    ''' %tuple(cca.cancorrs))
    nTicks = max(testcorrs[0].shape[0],testcorrs[1].shape[0])
    bmap1 =qualitative.Dark2_3
    plt.plot(np.arange(testcorrs[0].shape[0])+1, testcorrs[0], 'o', color = bmap1.mpl_colors[0])
    plt.plot(np.arange(testcorrs[1].shape[0])+1, testcorrs[1], 'o', color = bmap1.mpl_colors[1])
    plt.xlim(0.5, 0.5 + nTicks + 3)
    plt.ylim(0.0, 1.0)
    plt.xticks(np.arange(nTicks)+1)
    plt.xlabel('Dataset dimension')
    plt.ylabel('Prediction correlation')
    plt.title('Prediction accuracy')
    plt.legend(['Dataset 1', 'Dataset 2'])
    print('''The prediction accuracy for the first dataset is:\n
    Dimension 1: %.02f\n
    Dimension 2: %.02f\n
    Dimension 3: %.02f\n
    '''% tuple(testcorrs[0]))
    print('''The prediction accuracy for the second dataset is:\n
    Dimension 1: %.02f\n
    Dimension 2: %.02f\n
    Dimension 3: %.02f\n
    '''% tuple(testcorrs[1]))
    #cca.fit(data1,data2)
    #X_c, Y_c = cca.transform(data1,data2)
    return None







    









     





