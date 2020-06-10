import math
import numpy as np
from scipy import signal


#Function to calculate the mean along the n_ch axis that along the rows: Courtesy Wenjun Jia
def zero_mean(data, axis=0):
    mean = data.mean(axis=1, keepdims=True)# keep dimension parameter preserves the original dimension after averaging
    return data - mean #We can subtract as we preserved the dimension

#Function for spataial correlation: Courtesy Wenjun Jia
def spatial_correlation(averaged_v1, averaged_v2, std_v1, std_v2, n_ch):
    correlation = np.dot(averaged_v1, averaged_v2) / (n_ch * np.outer(std_v1, std_v2))
    return correlation



#Function to back fit the microstate maps on the raw data: Courtesy Wenjun Jia
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





def backFit(sigDiffMapLabel):
    # sigDiffMapLabel will be hash map having key as label and value as variable: maps from k-means 
    maps = sigDiffMapLabel['label']


    label = fit_back(data, maps, distance= 10, n_std=3, polarity=False)


    return None



