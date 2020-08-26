FULL STRUCTURE OF THE MS_RS EEG muscle artifacts detection and removal ALGORITHM:
MAIN:
main folder e dhuklam:
Created a configuration file: ConfigFile.py
file rakbo artifact detection and removal source path or file path
function thakbe .py script thakbe string nilo filepath nilo
_init_.py file diya filepath pass korbo artifact detection and removal function
----------------------------------------------------
artifact_detection:

scripty name: eeg_artifact_analyzer

function name: detect_artifact: for now: filepath file validation korbo er vitore 
proprocess k call +power_analysis k call funtion name; remove_artifact: 
for now: validate input parameter, input from power_analysis kaj korbe 
microstate+randomization+stat call hobe function name: 
same jinish analyze_eeg_artifact:input hishebe filepath nibe 
detect_artifact and remove_artifact call korbo

Same jinish: artifact_detection: function: input: filepath, optional parameters: 
backfit = False/True, interpolate = True/False, visualize = True/False,  
validation/comparison = True/False
		
----------------------------------------------------
Preprocess call hobe(done)
Band pass filtering(done)
Notch filtering(done)
Setting the average reference(done)
----------------------------------------------------
Simulation of EEG data from preprocessed raw data(done)
Power analysis call hobe on preprocessed raw data(done)
Power analysis on the simulated EEG data
Detection of threshold based on: "Simulated EEG data","preprocessed raw" 
and "simulated from MATLAB"EEG data. 
Setting a threshold of mean(done), median or mean + one std
Format the data for microstate analysis(done)
----------------------------------------------------
microstate analysis: script MicrostateAnalyzer.py
Step 1: Randomly assign data from n subjects into 2 sets: Trianing set and test set.(done)
An example is say we have 4 subjects: So set 2 subjects data on training and rest 2 on 
the test set. etc etc. With 50% test set and 50% training set.(done)
Dividing data in to test data and training data. 50% training data and 50% test data(done)
Step 2: Choosing optimal number of clusters: running microstate analysis varying n_maps 
parameter from 3 to 20(done) 
Step 3: Generation of optimal number of microstate classes(done)
Step 4: Application to the artifactual data on the basis of regions (done)
application to the raw data on the basis of regions(done)
Step 5: Application on the average data of subjects obtained condition/group/region wise 
both before and after random shuffling of the data of the subjects
Step 6: Finding the quantifiers like onset(done), offset(done),
count of time ponts(done), gfp, duration etc for each microstate clusters or 
classes or maps  
----------------------------------------------------

Randomization statistical analysis:
Calculate the grand mean ERP maps for every group or region:
Group/Region: Left Frontalis, Right Frontalis, Left Temporalis, Right Temporalis, 
Combined-all
Condition_name/Factor: contaminated, non-contaminated
----------------------------------------------------
Single subject data: Dividing the data into 10 2-second epochs(done)
Step 1: Take the average of the 10 2-second epochs
Step 2: Determine optimal number of microstate classes as mentioned in the paper(done)
Step 3: Quantifiers of classes are calculated and varience is obtained(done)
Step 4: Now shuffle the 10 epochs and do the steps 1,2,3
Step 5: Repeat step 4 1000/5000 times 
Just take the difference between maps of two conditions.(done)
Subtract mean from the result
Calculate the std you get GFP:
Now shuffle the subject data radomly group wise:
repeat steps again get GFP:
Test for null hypothesis probability.(done)
For each microstate class: 
Computation of the parameters onset and offset latency, AUC, center of gravity, 
duration, mean GFP
Onset = 
AUC = sum(GFP_peaks) on those time points which are assigned to the given class.
AUC er jonno label parameter theke label count kore corresponding gfp_peak parameter 
er sum nite hobe. Temporal parameter: center of gravity**
-----------------------------------------------------
Mulitple subject data analysis:
Step 1: First collect some data from some subjects condition or group wise: 
		Like Faces1.bdf, Faces2.bdf etc etc
Step 2: Next take the average of the data
Step 3: Calculate the optimal number of clusters using microstate analyser script.
Step 4: Calculate the quantifiers of the microstate classes/clusters and 
take the varience!
Step 5: Next shuffle the data of all the subjects factor/group/condition wise randomly
Step 6: Repeat steps 2,3,4 1000/5000 times
Step 7: Count cases out of 1000/5000, where random values in step 6 were 
equal or greater than value obtained in step 4
  
Use shuffle data(done), tanova, permutation2, permutation3 functions of Wenjun
Follow Wenjun's code structure for calculating the map parameters.
Now at first getting the grand mean data from the subjects accross subjects and 
All same steps in the single subject analysis 
-----------------------------------------------------
Backfit:
Signinficantly different Microstate class or map is obtained(done) then back fitted 
on the preprocessed raw data using the time points traced from labels parameter to 
retain the non-contaminated EEG segments. Maps parameter can also be used to combine the 
Signinficantly different Microstate maps.
Draft code is ready(done) 
-----------------------------------------------------
Interpolate:
Same like backfit only not significantly different maps are chosen from the maps variable.
Draft code is ready (done)
-----------------------------------------------------
Validation and comparison  
Comparison between simulated data and artifact removed data from the proposed method
Comparison with artifact removed data by ICA-MARA and the proposed method
Comparison of the performance between the ICA-MARA method and the proposed method 
applied on the simulated. EEG data produced from the real experimental data 
-----------------------------------------------------
Denoised EEG data quality testing
Checking correlation of artifact removed data with the contaminated data segments.
Checking correlation also with the EOG data
Checking some parameters like signal to noise ratio, signal to channel ratio, 
other parameters explained in the HAPPE paper.(harvard paper on preprocessing) 
-----------------------------------------------------
May 9, 2020
1) logging library in python
2) Debug mode 
3) Task: .py file gula boro hater hobe title case hobe
4) Aki script er function camel case hobe
5) NumPy library naming convention: Folder name theke no. of 
6) filepath_validation script:
Recap: script er name title case: Main
function er name: detectEegArtifact
variable same hobe
contant uppercase and _ diye

May 10, 2020
Common kaj akta function nibo jeta onno channel er upor depend kore nah

May 18, 2020
Finally finished Power Analysis code refactoring. Thanks to Shiblu vai!!!

May 19, 2020
Noting the tasks
i. Application of microstate analysis to get optimal number of microstate classes. 
Use cross validation techinque.
ii. Use 50% data as training set and 50% as test data
iii. Then generate the parameters for Randomization statistics
iv. Comparison of the microstate classes and their assignment between the 
factor levels or to test differences between groups for statistical significance
v. Quantifiers for each microstate class are: duration, area under the curve (AUC), 
mean GFP as global parameters
vi. Robust parameter is center of gravity. Optional onset offset of microstate classes 

May 20, 2020
Updated the configure file

May 21, 2020
Research meeting at 1 pm and report task.
   
May 22, 2020
Thesis writing chapter 4

May 23, 2020
Thesis writing chapter 5

May 24, 2020
Thesis writing chapter 5

May 25, 2020
Thesis writing chapter 5

May 26, 2020
Coding architecture for randomization statistical analysis

May 27, 2020
Thesis writing chapter 4,5,6
Report for meeting 

May 28, 2020
Research meeting at 11 am and 1pm and task reporting.
Thesis chapter 1 editing

May 29, 2020
Thesis chapter 4 editing, Quantifiers of microstate class coded

May 30, 2020
Improved preprocessing code

May 31- June 17
Improved and refactored the first four parts: Preprocessing, Microstate analysis,
Randomization statistics, Backfit. Ready for testing.

Last meeting: 11th June 2020 
search microstate analysis
search randomization stat analysis 
recent years

June 22, 2020:
Figured out the randomization statistical analysis using map difference concept.
We find the number of optimal clusters.
Next condition wise we find the optimal map templates!
Those templates will have a difference. The difference will be for channels. So we will 
get a varience channelwise. This varience will be available for each microstate class or 
map. That varience can be used to conduct ANOVA, other statistical test to find out the 
difference between the microstate classes that come from two condition.
Other microstatic parameters like duration, frequncy of occurance, 

JULY 29, 2020: Session at 11:00 PM
Task 1: k-means optimize korte faile hole.. proper threading
Task 2: Asyncio python dekho with return value ber koro fast****
Task 3: Multiprocess with function return value.... cpu er kaj

parameterize korte hobe. 







  