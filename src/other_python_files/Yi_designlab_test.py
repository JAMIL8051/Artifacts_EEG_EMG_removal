import numpy as np
import mne
import tkinter as tk
from tkinter import filedialog
from sklearn.decomposition import FastICA


root = tk.Tk()
root.withdraw()

print("Please select the data file")
path_data = filedialog.askopenfilename()
print("Please select the channel location file")
path_loc = filedialog.askopenfilename()





def read_data(path):
    with open(path, 'r') as f:
       lines = f.readlines()
    matrix = []
    for line in lines:
        res = []
        temp = line.split(' ')
        for num in temp:
            if num:
                res.append(float(num))
        matrix.append(res)
    return np.asarray(matrix)

def read_pos_data(path=None):
    if path is None:
        path = 'Cap63.locs'
    with open(path, 'r') as f:
        lines = f.readlines()
    res = []
    for line in lines:
        temp = line.split('\t')
        row = []
        if len(temp) > 1:
            row.append(float(temp[1].strip()))
            row.append(float(temp[2].strip()))
            row.append(temp[3].strip())
        if row:
            res.append(row)
    return np.asarray(res)


#path_data = "F:\\Dou\\Stress Science\\interface\\test.mul"
#path_loc = "F:\\Dou\\Stress Science\\interface\\Cap63.locs"

data = read_data(path_data)
loc = read_pos_data(path_loc)

print("Input Data Size {}".format(data.shape))

# ICA
ica = FastICA()
s = ica.fit_transform(data)     # Reconstruct signals
print("Reconstructed Data Size {}".format(s.shape))

n_channels = 63
sampling_rate = 500
info = mne.create_info(ch_names=loc[:, 2].tolist(), sfreq=sampling_rate, ch_types=['eeg']*63)
print(info)

revdata = data.T
print(revdata.shape)

# plot PSD
raw = mne.io.RawArray(revdata, info)
print(raw)
raw.plot_psd()