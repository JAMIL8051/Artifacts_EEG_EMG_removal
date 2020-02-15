import math
import scipy
import scipy.spatial.distance as dist
from scipy.stats import multivariate_normal, pearsonr, f_oneway
import numpy as np
import mne
import sklearn.metrics
#import tkinter as tk
#from tkinter import filedialog
#import EegPreprocessor as preprocessor
#import eeg_visualizer as plotter
#import microstates
#import MicrostateAnalyzer as ms_analyze
#import pca_ica_gfp_freq_bands as analysis gfp_analysis
import scratch as testing
import argparse, os, sys, time
from matplotlib import pyplot as plt
import pandas as pd



#Step1: Importing EDF data files
chs, fs, data_raw = testing.read_edf("test.edf") 
chs_bads, fs_bads, data_raw_bads = testing.read_edf('C:/projects/eeg_microstates/src/2019-05-03-suject-01-192sec_bad.edf')
chs_grp1, fs_grp1, data_raw_grp1 = testing.read_edf('C:/projects/eeg_microstates/src/2019-05-03-suject-01-Channel_grp1_6sec.edf')
chs_grp2, fs_grp2, data_raw_grp2 = testing.read_edf('C:/projects/eeg_microstates/src/2019-05-03-suject-01-Channel_grp2_6sec.edf')

#Band pass filtering
data = testing.bp_filter(data_raw, 1, 35, fs) 
data_bads = testing.bp_filter(data_raw_bads, 1, 35, fs_bads)
data_grp1 = testing.bp_filter(data_raw_grp1, 1, 35, fs_grp1)
data_grp2 = testing.bp_filter(data_raw_grp2, 1, 35, fs_grp2)

# Optimal No. of microstate clusters or maps 4 or 6
n_maps = 6

# Modified k-means algorithm
maps, x, gpf_peaks, gev, cv =testing.kmeans(data,n_maps,n_runs =10, maxerr=10e-6,maxiter=500)
maps_bads, x_bads, gfp_peaks_bads, gev_bads, cv_bads = testing.kmeans(data_bads, n_maps, n_runs = 10, maxerr = 10e-6, maxiter = 500 )
maps_grp1, x_grp1, gfp_peaks_grp1, gev_grp1, cv_grp1 = testing.kmeans(data_grp1, n_maps, n_runs = 10, maxerr = 10e-6, maxiter = 500 )
maps_grp2, x_grp2, gfp_peaks_grp2, gev_grp2, cv_grp2 = testing.kmeans(data_grp2, n_maps, n_runs = 10, maxerr = 10e-6, maxiter = 500 )
print('\n\t Microstate Analysis succesful')

#Calculation of GEV for each groups on basis of gfp peaks:
#Basic statistics
#GEV: Global explained varience:
total_gev = gev.sum()
total_gev_bads = gev_bads.sum()
total_gev_grp1 = gev_grp1.sum()
total_gev_grp2 = gev_grp2.sum()

print("Global explained varience (GEV) of bad channels group  per map:"+str(gev_bads))
print("Total GEV for bad channels group: {:.2f}".format(gev_bads.sum()))
print("Global explained varience (GEV) of channels group 1  per map:"+ str(gev_grp1))
print("Total GEV for channels group 1: {:.2f}".format(gev_grp1.sum()))
print("Global explained varience (GEV) of channels group 2  per map:"+ str(gev_grp2))
print("Total GEV for channels group 1: {:.2f}".format(gev_grp2.sum()))

#PPS (Per second gfp peaks)
pps = len(gpf_peaks)/(len(x)/fs)
pps_bads = len(gfp_peaks_bads)/(len(x_bads)/fs_bads)
print("GFP peaks per second for bad channels group: {:.2f}".format(pps_bads))
pps_grp1 = len(gfp_peaks_grp1)/(len(x_grp1)/fs_grp1)
print("GFP peaks per second for channels group 1: {:.2f}".format(pps_grp1))
pps_ch_grp2 = len(gfp_peaks_grp2)/(len(x_grp2)/fs_grp2)


#Class wise 1 factor(1 way) Topographic ANOVA(TANOVA)

fvalue_bad, pvalue_bad = f_oneway(maps_bads[0], maps_bads[1], maps_bads[2], maps_bads[3])
print(fvalue_bad, pvalue_bad)

fvalue_grp1, pvalue_grp1 = f_oneway(maps_grp1[0], maps_grp1[1], maps_grp1[2], maps_grp1[3])
print(fvalue_grp1, pvalue_grp1)

fvalue_grp2, pvalue_grp2 = f_oneway(maps_grp2[0], maps_grp2[1], maps_grp2[2], maps_grp2[3])
print(fvalue_grp2, pvalue_grp2)


#Parametric Statistical Hypothesis Test

#Analysis of Varience Test (ANOVA)
for j in range(0,n_maps):
    for i in range(0,n_maps):
        stat_anova_bads, p_anova_bads = f_oneway(maps_bads[j],maps_grp1[i],maps_grp2[i])
        print('stat=%.3f, p=%.3f' %(stat_anova_bads,p_anova_bads))
        if p_anova_bads>0.05:
            print('Probably the same distribution')
        else:
            print('Probably different distributions')

#ANOVA for Group 1 channels:
stat_anova_grp1, p_anova_grp1 = f_oneway(maps_grp1[0],maps_grp1[1],maps_grp1[2],maps_grp1[3])
print('stat=%.3f, p=%.3f' %(stat_anova_grp1,p_anova_grp1))
if p_anova_grp1 > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')

#ANOVA for Group 2 channels:
stat_anova_grp2, p_anova_grp2 = f_oneway(maps_grp2[0],maps_grp2[1],maps_grp2[2],maps_grp2[3])
print('stat=%.3f, p=%.3f' %(stat_anova_grp2,p_anova_grp2))
if p_anova_grp2 > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')



#ANOVA of Microstate classes: A, B, C, D
stat_A, p_value_A = f_oneway(maps_bads[0],maps_grp1[0],maps_grp2[0])
print('stat=%.3f, p=%.3f' %(stat_A,p_value_A))
if p_value_A > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')

stat_B, p_value_B = f_oneway(maps_bads[1],maps_grp1[1],maps_grp2[1])
print('stat=%.3f, p=%.3f' %(stat_B,p_value_B))
if p_value_B > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')

stat_C, p_value_C = f_oneway(maps_bads[2],maps_grp1[2],maps_grp2[2])
print('stat=%.3f, p=%.3f' %(stat_C,p_value_C))
if p_value_C > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')

stat_D, p_value_D = f_oneway(maps_bads[3],maps_grp1[3],maps_grp2[3])
print('stat=%.3f, p=%.3f' %(stat_D,p_value_D))
if p_value_D > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')

#Basic Statistical analysis of the two map groups:Bad channels and Group 1 channels
#Assumption of independence and t-test between maps and unpaired t-test

#Mean of the maps
mean_maps_bads = np.mean(maps_bads)
mean_maps_grp1 = np.mean(maps_grp1)

#Variance
var_maps_bads = np.var(maps_bads)
var_maps_grp1 = np.var(maps_grp1)

#Covarience 
cov_maps_bads = np.cov(maps_bads, y= None, rowvar = True)
cov_maps_grp1 = np.cov(maps_grp1, y = None, rowvar = True)


#Mean difference 
mean_diff = abs(mean_maps_bads-mean_maps_grp1)

#No of observations(no. of channels)
n1 = 12
n2 = n1

#For the Calculation of t-value
noise = math.sqrt((var_maps_bads/n1)+(var_maps_grp1/n2))

#Student t-value
t_value = mean_diff/noise

#Student t-test: Degree of freedom df
df =n1+n2-2
statistic, p_value = scipy.stats.ttest_ind_from_stats(np.mean(maps_bads[0]),np.std(maps_bads[0]), n1, np.mean(maps_grp1[0]), np.std(maps_grp1[0]), n2, equal_var = False)
statistics1, p_value1 = scipy.stats.ttest_ind(np.mean(maps_bads[0]), np.mean(maps_grp1[0]),axis = 0, equal_var = False, nan_policy ='propagate')

#Basic statistcs:
#Empirical label distribution
p_hat_bads = testing.p_empirical(x_bads, n_maps)
p_hat_grp1 = testing.p_empirical(x_grp1, n_maps)

#Transition matrices
T_hat_bads = testing.T_empirical(x_bads, n_maps)
T_hat_ch_grp1 = testing.T_empirical(x_grp1, n_maps)


#Pearson's Correlation Coefficient test
for i in range(0, n_maps):
    for j in range(0, n_maps):
        stat_pearson, pearson_p_value = pearsonr(maps_bads[i], maps_grp1[j])
        print('stat=%.3f, p=%.3f'%(stat_pearson,pearson_p_value))
        if pearson_p_value > 0.05:
            print("Probably independent: The bad channels map:{:1} and group1 channels map: {:1}".format(i,j))
        else:
            print("Probably dependent: The bad channels map:{:1} and group1 channels map: {:1}".format(i,j))



#Calculation of cosine similarities among the maps of bad channels group and group 1
for i in range(0,n_maps):
    for j in range(0,n_maps):
        #euclid_dis = dist.euclidean(bad_ch_maps[i],ch_grp1_maps[j])
        #print("Euclidean distance between bad channels map:{:.1f} and group1 channels map: {:.1f} is {:.6f}".format(i,j,euclid_dis))
        #manhattan_dis = dist.cityblock(bad_ch_maps[i],ch_grp1_maps[j])
        #print("Manhattan distance between bad channels map:{:.1f} and group1 channels map: {:.1f} is {:.6f}".format(i,j,manhattan_dis))
        #Cheby_dist = dist.chebyshev(bad_ch_maps[i],ch_grp1_maps[j])
        #print("Chebyshev distance between bad channels map:{:.1f} and group1 channelsmap: {:.1f} is {:.6f}".format(i,j, Cheby_dist))
        #canberra_dist = dist.canberra(bad_ch_maps[i],ch_grp1_maps[j])
        #print("Canberra distance between bad channels map:{:.1f} and group1 channelsmap: {:.1f} is {:.6f}".format(i,j,canberra_dist))
        cosine_dist_scipy = dist.cosine(maps_bads[i],maps_grp1[j])
        cosine_similarity_scipy = 1 - cosine_dist_scipy
        print("Cosine similarity using Scipy between bad channels map:{:1} and group1 channelsmap: {:1} is {:.6f}".format(i,j,cosine_similarity_scipy))
       

cosine_similarity = sklearn.metrics.pairwise.cosine_similarity(maps_bads,maps_grp1)
print(cosine_similarity)

#Printing the microstates classes
#channels, locs = testing.read_xyz('cap.xyz')
#for i, map in enumerate(maps):
#    plt.figure(figsize=(2 * len(maps),5))
#    plt.subplot(2, len(maps), i+1)
#    plt.title("Maps: {}".format(i))
#    mne.viz.plot_topomap(map, pos = locs[:, :2])

#Printing the microstates classes
#channels1, locs1 = testing.read_xyz('biosemi64_bad_ch_pos.xyz')
#for i, map in enumerate(maps_bads):
#    plt.figure(figsize=(2 * len(maps_bads),5))
#    plt.subplot(1, len(maps_bads), i+1)
#    plt.title("Bad Channels Map: {}".format(i))
#    mne.viz.plot_topomap(map, pos = locs1[:, :2])


#channels, locs = testing.read_xyz('biosemi64_ch_grp1_pos.xyz')
#channels2, locs2 = testing.read_xyz('biosemi64_ch_grp2_pos.xyz')

#for i, map in enumerate(maps_grp1):
#    plt.figure(figsize=(2* len(maps_grp1),5))
#    plt.subplot(1, len(maps_grp1), i+1)
#    plt.title("Channels Group 1 Map: {}".format(i))
#    mne.viz.plot_topomap(map, pos = locs[:, :2])

#for i, map in enumerate(ch_grp2_maps):
#    plt.figure(figsize=(2* len(ch_grp1_maps),5))
#    plt.subplot(1, len(ch_grp1_maps), i+1)
#    plt.title("Channels Group 1 Map: {}".format(i))
#    mne.viz.plot_topomap(map, pos = locs2[:, :2])

     


















#for i in range(1):
#    print(__doc__)
    
    
#    DEFAULT_EOG = ['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']
    
#    def get_raw_data_file_path():
#        #root = tk.Tk()
#        #root.withdraw()
#        #data_file_path  = filedialog.askopenfilename()
#        return 'C:/projects/eeg_microstates/src/test.edf'
    
#    #Channel position function for Biosemi machine
#    def load_biosemi_montage(string = 'biosemi64'):
#        montage = mne.channels.make_standard_montage(string)
#        return montage
    
#    #Loading the raw data
#    def load_raw_data(filepath, montage, eog = DEFAULT_EOG):
#        raw = mne.io.read_raw_edf(filepath, montage, eog, preload =True)
#        return raw
    
    
#    def preprocess_raw_data():
#        print("Please enter the raw data file")
#        filepath = get_raw_data_file_path()
        
#        montage = load_biosemi_montage()
#        raw = load_raw_data(filepath, montage)
        
# Using the average EEG reference
#        raw.set_eeg_reference('average')
# High pass filtering of data
#        raw.filter(0.2, None)
#        return raw
#    raw = preprocess_raw_data()
    
#    data = raw.get_data()
#    data = np.resize(data,(30,))

    
    
    
    
#    bad_channels_1st_32 = ['FT7', 'FC5', 'P9', 'FC3', 'POz', 'PO3', 'Iz'] 
#    bad_channels_2nd_32 = ['Fp2', 'FT8', 'C4', 'AF8', 'F8']
#    
#    raw_copy1 = raw.copy()
#    raw_copy2 = raw.copy()
#    
#    
#
#    raw_pick_bad_channels_1st_32 = raw_copy1.pick_channels(ch_names = bad_channels_1st_32)
#    raw_pick_bad_channels_2nd_32 = raw_copy2.pick_channels(ch_names = bad_channels_2nd_32)
#    
#    data_bad_channels = raw_pick_bad_channels_1st_32.get_data()
#    data_bad_channels_tr = data_bad_channels.transpose()
#    data_bad_channels = np.resize(data_bad_channels,(len(bad_channels_1st_32),len(data_bad_channels_tr)))
#    
#    data_bad_channels_2 = raw_pick_bad_channels_2nd_32.get_data()
#    data_bad_channels_tr_2 = data_bad_channels_2.transpose()
#    data_bad_channels_2 = np.resize(data_bad_channels_2,(len(bad_channels_2nd_32),len(data_bad_channels_tr_2)))
#    
#    maps_bad_channels_1st_32, segmentation_bad_channels_1st_32 = microstates.segment(data_bad_channels, n_states= 4, n_inits = 300)
#    microstates.plot_maps(maps_bad_channels_1st_32, raw_pick_bad_channels_1st_32.info)
#    
#    maps_bad_channels_2nd_32, segmentation_bad_channels_2nd_32 = microstates.segment(data_bad_channels_2, n_states= 4, n_inits = 300)
#    microstates.plot_maps(maps_bad_channels_2nd_32, raw_pick_bad_channels_2nd_32.info)
    
    
    
    
    
    
    