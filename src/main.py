import tkinter as tk
from tkinter import filedialog
import numpy as np
import mne
import microstates
import os
from pathlib import Path


#Function to read the MUL data file
def read_data(path):
    with open (path,'r') as f:
        lines = f.readlines()
        matrix = []
        for line in lines:
            res=[]
            temp = line.split(' ')
            for num in temp:
                if num:
                    res.append(float(num))
            matrix.append(res)
        return np.asarray(matrix)


#Tkinter package for user selection of files: DATA and Channel Location file
root = tk.Tk()
root.withdraw()

print("Please select the data file")
data_file_path  = filedialog.askopenfilename()

print("Please select the channel location file")
channel_file_path = filedialog.askopenfilename()


# "kind" and "path" variables for mne.channels_read_montage function 
p = Path(channel_file_path)
k = p.parts[-1]
d = k.find('.')
kind = k[0:d]

f = p.parts
f=f[0:len(f)-1]
path = os.path.join(*f) 
    


#Loading of the data: Prefereable format .MUL

data1 = read_data(data_file_path)
data = np.resize(data1,(63,30000))

#The name of the channels. It can be modified as desired. By default 63
ch_names = ['FP1','Fz','F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3','T7','TP9',
            'CP5','CP1','Pz','P3','P7','O1','Oz','O2','P4','P8','TP10','CP6',
            'CP2','C4','T8','FT10','FC6','FC2','F4','F8','FP2','AF7','AF3',
            'AFz','F1','F5','FT7','FC3','FCz','C1','C5','TP7','CP3','P1','P5',
            'PO7','PO3','POz','PO4','PO8','P6','P2','CPz','CP4','TP8','C6',
            'C2','FC4','FT8','F6','F2','AF4','AF8']

#In the unit parameter "cm"/"m" can be given
montage = mne.channels.read_montage(kind='Cap63',ch_names = ch_names, 
                                    path = path, unit='cm', transform=False)
#creating the channel info instance
info = mne.create_info(ch_names = ch_names, sfreq=250,ch_types ='eeg', 
                       montage = montage, verbose = None)
#Creating the raw instance of the data
raw = mne.io.RawArray(data,info,first_samp= 0, verbose = None)


#"""" OPTIONAL PARTS Raw data visualization """
#Auto scaling option
scalings ='auto'
raw.plot(n_channels = 63, scalings=scalings,title='Auto-scaled Data from arrays',
         show=True,block=True)

#EEG Microstates
# Segment the data in number of microstates
n_states = int(input("Please provide the number of Microstates: "))
if n_states <2 :
    print("The number of microstates must be equal greater than or equal to 2" )

n_inits = int(input("Please give the number of random initializations to use for the k-means algorithm: "))

maps, segmentation = microstates.segment(raw.get_data(), n_states= n_states, n_inits = n_inits)

# Plot the topographic maps of the microstates and the segmentation
print(" Visualizing the topographical maps of the EEG Micrsotates ")

microstates.plot_maps(maps, raw.info)
#Plotting the segementation for first 600 time samples
microstates.plot_segmentation(segmentation[:600], raw.get_data()[:, :600],
                              raw.times[:600])
