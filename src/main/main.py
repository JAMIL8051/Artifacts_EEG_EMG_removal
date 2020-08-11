import Configuration
import EegArtifactAnalyzer
from tkinter import filedialog
import tkinter as tk


"""
@Author: Jamil Reza Chowdhury
Department of Electrical and Computer Engineering 
Concordia University, Montreal, QC Canada
"""

"""The main function to call for the analysis and removal of EMG artifacts from EEG 
recordings using micrsotates and randomization statistics. The file uses the module
eeg_artifact_analyzer and then calls the function artifact_detection_and_removal 
function.
The basic steps of this main file are:
1. Takes the fully specified absolute filepath as input from the user
Calls the artifact_detection_and_removal function.
"""


#Simple take the filepath from the user and calls the eeganalyzer to detecta nd remove the artifacts 
#and finally visualize the clean data using the visualizer.

def main():
    print("Please enter the full path of the raw data file. The format should be bdf or edf or vhdr")
    root = tk.Tk()
    root.withdraw()
    filepath  = filedialog.askopenfilename()
    print(filepath)
    #filepath = 'C:/Users/J_CHOWD/Desktop/EEG_microstate_analysis_papers/TestDataN-BackLucas/2019-05-03_001.bdf'
    print('Initializing the detection and removal process')
    trainDataPath = Configuration.defaultTrainDataFolder()
    results = EegArtifactAnalyzer.detectAndRemoveEegArtifact(filepath, trainDataPath, backfit= True, 
                                                             interpolate= True, comparison = True, visualize = True)

    
    return None

main()


