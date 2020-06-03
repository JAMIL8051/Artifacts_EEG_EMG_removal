import Configuration
import numpy as np
import mne


# Function for Power_analysis of preprocessed EEG raw object
def finalBandPower(raw, N, fs, tmin = None, tmax= None, epoch_time = 2):
	"""
		raw = Mne.raw object
		N = No. of samples for FFT algorithm
		fs = sampling frequecy 
		tmin = starting time of the trials
		tmax = end time of the trials
		epoch_time = duration of the epoch by default 2 seconds
		"""
	#raw = raw.pick(picks=channel)
	
	if (tmin == None and tmax == None):
		n_epochs = 100
	else:
		n_epochs = int((tmax-tmin)/epoch_time)
	
	data = raw.get_data()
	data = data/1e-06
		
	#Normalizing the data before power calculations
		
	n_channels = len(raw.ch_names)
	epoch_power_45_70Hz = np.zeros((n_channels, n_epochs, 1), dtype = 'float')
	
	for i in range(n_channels):
		for epoch in range(n_epochs):
			# A small investigation leads to the fact that I had to remove the 
			# first sample of from evey channel as the value was very 
			# very close zero.  
			epoch_data = data[i, ((N*epoch)+1):(N*(epoch+1)+1)] #Formation of 2s epoch data with N samples
			epoch_data = np.fft.fft(epoch_data, N)
				
			#calculation of k in the formula
			k_lower = int(45*(N/fs)) # Forcing these to integers otherwise index error problem will pop up 
			k_upper = int(70*(N/fs))
			temp = np.zeros((k_upper-k_lower),dtype='float')
			for k in range(k_lower, k_upper):
				val = (np.abs(epoch_data[k_lower])) ** 2 + (np.abs(epoch_data[N - k_lower])) ** 2
				temp[k-90] = val
			
			power_45_70Hz = (1/N**2)*sum(temp)
			epoch_power_45_70Hz[i, epoch, :] = power_45_70Hz
			
	return epoch_power_45_70Hz, data*1e-06


# Function to calculate the power of the EEG data epochs from the preprocessed Raw object from 
# the raw EEG data
def bandPower(raw, channels):
	filteredRaw = raw.copy().pick(picks = channels)
	power45_70Hz = finalBandPower(filteredRaw, N = 1024, fs = raw.info['sfreq']) 
	return power45_70Hz


# Function for the detection of contaminated data
def detectContaminatedData(raw, channelNames, threshold_val):
	power, data = bandPower(raw, channelNames)
	#Allocating memory for the channel and epoch indices array for the primary channels data			
	contaminatedData = np.zeros((len(channelNames),100,1024),dtype='float')

	for i in range(len(channelNames)):
		for epoch in range(100):
			if power[i,epoch] > threshold_val:
				contaminatedData[i,epoch,:] = data[i, (1024*epoch)+1:(1024*(epoch+1))+1]
	
	return contaminatedData


# The main function to identify the EMG contaminated data:
def identifyArtifacts(raw):
	"""
	input parameter: raw object only from MNE library Python
	"""
	# getting the channels map as dictionary from config file region as a list
	channelNamesMap, region = Configuration.channels()
	channelNamesPrimary = []
	ch_names_list = []
	for key in channelNamesMap:
		if key != 'Fz':
			channelNamesPrimary.append(key)    
			ch_names_list.append(channelNamesMap[key])
		
	ch_names_combined = sum(ch_names_list, [])

	power, data = bandPower(raw, channelNamesPrimary)
	threshold = power.mean(axis = 1)
	
	artifactualData = {}
	finalEmgData = np.zeros((len(channelNamesPrimary), len(ch_names_list[0]), 100, 1024), dtype = 'float')
	
	for i in range(len(channelNamesPrimary)):
		primaryChannelName = channelNamesPrimary[i]
		for epoch in range(100):
			if power[i, epoch]>threshold[i]:
				contaminatedData = detectContaminatedData(raw, channelNamesMap[primaryChannelName], threshold[i])
				finalEmgData[i,:,:,:] = contaminatedData
				artifactualData[region[i]] = contaminatedData[contaminatedData != 0]

	
	dataWithArtifactsDetected = finalEmgData
	
	
	return dataWithArtifactsDetected, ch_names_combined, artifactualData 
	














