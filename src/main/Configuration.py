"""
Configuration file for analysis. This file follows the 'Biosemi64' cap electrode layout system. Anybody can compare other 
electrode layout can configure the same file for analysis.

This file is editable and reusable for conducting the whole analysis.
The first channelNamesMap is a python dictionary. It's keys are AF7, AF8, FT7, FT8 channels.
It is the mapping of all channels in the following regions: 'Left Frontalis', 'Right Frontalis', 
'Left Temporalis', 'Right Temporalis'

'Left Frontalis' region - channels- 'Fp1', 'AF3', 'F3', 'F5', 'F7'- Primary channel of interest AF7
AF7 channel is mostly contaminated by left frontalis brain muscle. 
'Right Frontalis' region - channels- 'Fp2', 'AF4', 'F4', 'F6', 'F8'- Primary channel of interest AF8
AF8 channel is mostly contaminated by right frontalis brain muscle.

'Left Temporalis region - channels- 'F7', 'F5', 'FC5', 'C5', 'T7'- Primary channel of interest FT7
FT7 channel is mostly contaminated by left temporalis brain muscle. 
'Right Temporalis region - channels- 'F8', 'F6', 'FC6', 'C6', 'T8'- Primary channel of interest FT8
FT8 channel is mostly contaminated by right temporalis brain muscle. 

The key Fz is the central channel from all the 4 regions(see 'Biosemi64' cap electrode layout system).
This has two channels Fz and Cz. Fz can be used for the calculating the detection threshold in the power analysis step. 
Cz is the reference channel by default.

The region variable shows the names of the regions in the brain which are highly prone muscle artifacts in EEG signals
due to frontalis and temporalis scalp muscle 
"""
def defaultTrainDataFolder():

    return  "C:/Users/J_CHOWD/Desktop/EEG_microstate_analysis_papers/TestDataN-BackLucas/*.bdf"


def channels():
    channelNamesMap = {'AF7': ['Fp1', 'AF3', 'F3', 'F5', 'F7'],'AF8':['Fp2', 'AF4', 'F4', 'F6', 'F8'],
                     'FT7': ['F7', 'F5', 'FC5', 'C5', 'T7'],'FT8': ['F8', 'F6', 'FC6', 'C6', 'T8'],
                    'Fz': ['Fz','Cz']}
    
    region = ['Left Frontalis', 'Right Frontalis', 'Left Temporalis', 'Right Temporalis']

    """ Option 1: I am using the configuration file with hard coded technique!"""
    channelNamesNoRepeat = {'AF7': ['Fp1', 'AF3', 'F3', 'F5', 'F7'],'AF8':['Fp2', 'AF4', 'F4', 'F6', 'F8'],
                     'FT7': ['FC5', 'C5', 'T7'],'FT8': ['FC6', 'C6', 'T8'],
                    'Fz': ['Fz','Cz']} 

   
    return channelNamesMap, region, channelNamesNoRepeat

# We used Bio-semi 64 electrode layout system. Other can be given like 10-20 system 10-10 sytem as well.
def channelLayout():
    return 'biosemi64'

# Names of channels for finding optimal clusters from raw data
def channelList():
    channelsOptimalCluster = ['Fp1', 'AF3', 'F3', 'F5', 'F7', 'Fp2', 'AF4', 'F4', 'F6', 'F8',
                              'FC5', 'C5', 'T7', 'FC6', 'C6', 'T8']
    return channelsOptimalCluster
# Parameters needed for conducting the microstate analysis.
 
"""
Parameter: used in microstate analysis in detecting the optimal number of clusters using the global explained varience
and cross validation techiques. Ref: A Student’s Guide to Randomization
Statistics for Multichannel Event-Related Potentials Using Ragu Marie Habermann1, Dorothea Weusmann1, Maria Stein2 and Thomas Koenig1* 1 Translational Research Center, Department of Psychiatric Neurophysiology, University Hospital of Psychiatry Bern, University of Bern, Bern, Switzerland, 2 Department of Clinical Psychology and Psychotherapy, University of Bern, Bern, Switzerland. doi: 10.3389/fnins.2018.00355   
"""
def numberOfCluster():
    return 9

"""
Parameter: used in microstate analysis in detecting the maxmimum average correlation for finding the optimal 
microstate model. Ref: A Tutorial on Data-Driven Methods for Statistically Assessing ERP Topographies
Thomas Koenig • Maria Stein • Matthias Grieder •
Mara Kottlow. Brain Topogr (2014) 27:72–83 DOI 10.1007/s10548-013-0310-1
"""    
def repetitionsCount():
    return 2


# Two condition of data: Contaminated data detected through power analysis and 
# rest of data being non-contaminated
def conditionForStatAnalysis():
    condition = ['contaminated','non-contaminated']
    return condition 

# Parameters for comparing the groups done region wise using the region parameter as mentioned early and 
# channel wise grouping: Primary channels only and channels combining the regions 
# excluding the primary channels

def groupForStatAnalysis():
    group =['region','channel']
    return group


