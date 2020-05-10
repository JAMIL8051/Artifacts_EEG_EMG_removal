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