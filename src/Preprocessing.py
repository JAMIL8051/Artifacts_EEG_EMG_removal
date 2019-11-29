import glob
import mne
import os
import pandas
import numpy as np

triggersLoadOne = [11, 41]

triggersLoadTwo = [21, 51]

triggersLoadThree = [31, 61]

workloadOne = np.zeros((1344, 1, 64, 576))

workloadTwo = np.zeros((1333, 1, 64, 576))

workloadThree = np.zeros((1344, 1, 64, 576))

numberChannels = 64 + 1

event_id = {'1-back_spatial': 10, '2-back_spatial': 20, '3-back_spatial': 30, '1-back_nonspatial': 40,
            '2-back_nonspatial': 50, '3-back_nonspatial': 60}

filePath = "/Users/lucashouse/Documents/CAE/*.bdf"

eegFiles = glob.glob(filePath)

# Load EEG file(s), filter, re-reference and find events

for file in eegFiles:
    raw = mne.io.read_raw_edf(file, preload=True)
    raw.filter(0.1, 100.0, fir_design='firwin')
    raw.notch_filter(np.arange(60, 241, 60), fir_design='firwin')
    raw.set_eeg_reference('average', projection=False)
    events = mne.find_events(raw)


k = 0

for trigger in triggersLoadOne:
    for event in events:
        if event[2] == trigger:
            data = raw.get_data()[0:64, event[0]:event[0] + int(4.5 * 512)]
            for x in range (1,5):
                part = data[0:64, 0 + (576 * (x - 1)):576 * x]
                workloadOne[k, 0, :, :] = part
                k = k + 1
                print(k)

k = 0

for trigger in triggersLoadThree:
    for event in events:
        if event[2] == trigger:
            data = raw.get_data()[0:64, event[0]:event[0] + int(4.5 * 512)]
            for x in range (1,5):
                part = data[0:64, 0 + (576 * (x - 1)):576 * x]
                workloadThree[k, 0, :, :] = part
                k = k + 1
                print(k)

workloadSamples = np.concatenate((workloadOne, workloadThree), axis=0)
np.save('./samplespPreprocessed', workloadSamples)



label = np.concatenate ((np.zeros(1344),np.ones(1344)),axis=0)
np.save('./labelsPreprocessed',label)



