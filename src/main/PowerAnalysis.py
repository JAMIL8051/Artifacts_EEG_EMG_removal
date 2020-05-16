import numpy as np
import mne


# Function for Power_analysis of preprocessed EEG raw object
def finalBandPower(raw,channel,N,fs,tmin=None, tmax=None,epoch_time=2):
	"""
		raw = Mne.raw object
		channel = no. of channels to include, type must be string
		N = No. of samples for FFT algorithm
		fs = sampling frequecy 
		tmin = starting time of the trials
		tmax = end time of the trials
		epoch_time = duration of the epoch by default 2 seconds
		"""
	raw = raw.pick(picks=channel)
	
	if (tmin == None and tmax == None):
		n_epochs = 100
	else:
		n_epochs = int((tmax-tmin)/epoch_time)
	
	data = raw.get_data()
	data = data/1e-06
		
	#Normalizing the data before power calculations
		
	n_channels = len(raw.ch_names)
	epoch_power_45_70Hz = np.zeros((n_channels, n_epochs, 1), dtype = 'float')
	
	for chan in range(n_channels):
		for epoch in range(n_epochs):
			# A small investigation leads to the fact that I had to remove the 
			# first sample of from evey channel as the value was very 
			# very close zero.  
			epoch_data = data[chan, ((N*epoch)+1):(N*(epoch+1)+1)] #Formation of 2s epoch data with N samples
			epoch_data = np.fft.fft(epoch_data, N)
				
			#calculation of k in the formula
			k_lower = int(45*(N/fs)) # Forcing these to integers otherwise index error problem will pop up 
			k_upper = int(70*(N/fs))
			temp = np.zeros((k_upper-k_lower),dtype='float')
			for k in range(k_lower, k_upper):
				val = (np.abs(epoch_data[k_lower])) ** 2 + (np.abs(epoch_data[N - k_lower])) ** 2
				temp[k-90] = val
			
			power_45_70Hz = (1/N**2)*sum(temp)
			epoch_power_45_70Hz[chan, epoch, :] = power_45_70Hz
			
	return epoch_power_45_70Hz, data*1e-06


# Function to calculate the power of the EEG data epochs from the preprocessed Raw object from
# the raw EEG data
def bandPower(raw,channels):
	filteredRaw = raw.copy().pick(picks=channels)
	power45_70Hz = finalBandPower(filteredRaw,channel = channels, N=1024,fs=512) 
	return power45_70Hz

# Function for the detection of contaminated channels and Epochs
def detectContaminatedEpochs(power45_70Hz,channelNames, threshold_val):
	#Allocating memory for the channel and epoch indices array for the primary channels data
	channelEpochs = np.zeros((len(channelNames),100),dtype='int')

	for channel in range(len(channelNames)):
		if power45_70Hz[channel,0] > threshold_val[channel]:
			channelEpochs[channel,0] = -1 # Setting -1 for the first epoch
			
		for epoch in range(1,100):
			if power45_70Hz[channel,epoch] > threshold_val[channel]:
				channelEpochs[channel,epoch] = epoch

	return channelEpochs


# Function to detect contaminated epochs in the child channels
def identifyArtifactsForChildChannels(power45_70Hz, channelNames, threshold):
	
	childEpochs = np.zeros((len(channelNames),100), dtype='int')
	for chan in range(len(channelNames)):
		if power45_70Hz[chan, 0] > threshold:
			childEpochs[chan, 0] = -1
		for epoch in range(1,100):
			if power45_70Hz[chan, epoch] > threshold:
				childEpochs[chan, epoch] = epoch

	return childEpochs

# Function to detect the contaminated data from contaminated channels and epochs
def detectContaminatedData(data, channelNames, chanelEpochs):
	contaminatedData = np.zeros((len(channelNames),100,1024),dtype='float')

	for channel in range(len(channelNames)):
		if chanelEpochs[channel,0] == -1:
			contaminatedData[channel,0,:] = data[channel, 1:1024+1]
		for epoch in range(1, 100):
			if chanelEpochs[channel,epoch]!= 0:
				contaminatedData[channel,epoch,:] = data[channel, (1024*epoch)+1:(1024*(epoch+1))+1]
	return contaminatedData




#Global function for identification of EMG artifacts by 
#power analysis in the 45-70Hz freuquency band.
def identifyArtifacts(raw):
	# Declaring the channel names variable for the analysis
	channelNamesPrimary = ['AF7','AF8','FT7','FT8']
	channelNamesAf7 = ['Fp1', 'AF3', 'F3', 'F5', 'F7']
	channelNamesAf8 = ['Fp2', 'AF4', 'F4', 'F6', 'F8']
	channelNamesFt7 = ['F7', 'F5', 'FC5', 'C5', 'T7']
	channelNamesFt8 = ['F8', 'F6', 'FC6', 'C6', 'T8']
	channelNamesFz = ['Fz']
	
	# Calling the bandpower function to calculate power and return data of channels
	powerPrimary, primaryData = bandPower(raw, channelNamesPrimary)
	af7Power45_70Hz, af7Data = bandPower(raw, channelNamesAf7)
	af8Power45_70Hz, af8Data = bandPower(raw, channelNamesAf8)
	ft7Power45_70Hz, ft7Data = bandPower(raw, channelNamesFt7)
	ft8Power45_70Hz, ft8Data = bandPower(raw, channelNamesFt8)
	fzPower45_70Hz, fzData = bandPower(raw, channelNamesFz)
	
	#Step 1 :Finding the threshold val without simulated data.
	#thresholdValPaper = power45_70Hz.mean(axis = 1)+1*power45_70Hz.std(axis = 1) 
	threshold_val = powerPrimary.mean(axis = 1)
	# Detecting the contaminated epochs

	primaryEpochs = detectContaminatedEpochs(powerPrimary, channelNamesPrimary,
												 threshold_val)

	# Will refactor call sequence
	childEpochsAf7 = identifyArtifactsForChildChannels(af7Power45_70Hz,  channelNamesAf7, primaryEpochs, 
													threshold_val[0])
	childEpochsAf8 = identifyArtifactsForChildChannels(af8Power45_70Hz,  channelNamesAf8, primaryEpochs, 
													threshold_val[1])
	childEpochsFt7 = identifyArtifactsForChildChannels(ft7Power45_70Hz,  channelNamesFt7, primaryEpochs, 
													threshold_val[2])
	childEpochsFt8 = identifyArtifactsForChildChannels(ft8Power45_70Hz,  channelNamesFt8, primaryEpochs, 
													threshold_val[3])

	
	#Storing the primary contaminated data obtained from contaminated epoch indices                
	primaryContaminatedData = detectContaminatedData(primaryData, channelNamesPrimary, channelEpochsPrimary) 
	Af7ContaminatedData = detectContaminatedData(af7Data, channelNamesAf7, childEpochsAf7) 
	Af8ContaminatedData = detectContaminatedData(af8Data, channelNamesAf8, childEpochsAf8) 
	Ft7ContaminatedData = detectContaminatedData(Ft7Data, channelNamesFt7, childEpochsFt7) 
	Ft8ContaminatedData = detectContaminatedData(Ft8Data, channelNamesFt8, childEpochsFt8) 
	
	# Getting the channel, epoch and time point indices of the non-zero contaminated data
	channels, epochs, times = np.nonzero(primaryContaminatedData != 0)
	channelsAf7, epochsAf7, timesAf7 = np.nonzero(Af7ContaminatedData != 0)
	
	# Setting the no. of channels and no. of time points 
	n_channels = len(channelNamesPrimary) + len(channelsAf7)
	n_time_points = int(len(times)/n_channels) + int(len(timesAf7)/len(channelsAf7))
	ContaminatedData = np.concatenate((primaryContaminatedData, Af7ContaminatedData), axis =0)
	dataWithArtifactsDetected = ContaminatedData[ContaminatedData != 0].reshape(n_time_points,
																					n_channels)

	
	sampling_rate = raw.info['sfreq']
	info = mne.create_info(ch_names = channelNamesPrimary, sfreq = sampling_rate, ch_types=['eeg']*n_channels)
	dataWithArtifactsDetectedRaw = mne.io.RawArray(dataWithArtifactsDetected.T, info)

	return dataWithArtifactsDetected, dataWithArtifactsDetectedRaw











##Using simulated data: Will do later
#sim_data_thres = threshold.sim_threshold(raw)

##Step 2 :Finding the channel epoch indices which are greater than threshold
#chan_epoch_indice_primary = np.zeros((len(primary_raw.ch_names),100),dtype='int')

#for chan in range(len(primary_raw.ch_names)):
#	for epoch in range(50):
#		if primary_power_45_70Hz[chan,epoch] > threshold_val:
#			chan_epoch_indice_primary[chan,epoch] = epoch 

#chan_epoch_indices_AF7 = np.zeros((len(AF7_raw.ch_names),50),dtype='int')
#chan_epoch_indices_AF8 = np.zeros((len(AF8_raw.ch_names),50),dtype='int')
#chan_epoch_indices_FT7 = np.zeros((len(FT7_raw.ch_names),50),dtype='int')
#chan_epoch_indices_FT8 = np.zeros((len(FT8_raw.ch_names),50),dtype='int')


#for idx in chan_epoch_indice_primary:
#	if  chan_epoch_indice_primary.any() == 0:
#		for chan in range(len(AF7_raw.ch_names)):
#			for epoch in range(50):
#				if AF7_power_45_70Hz[chan,epoch] >threshold_val[0]:
#					chan_epoch_indices_AF7[chan,epoch] = epoch
#	elif chan_epoch_indice_primary.any() == 1:
#		for chan in range(len(AF8_raw.ch_names)):
#			for epoch in range(50):
#				if AF8_power_45_70Hz[chan,epoch] >threshold_val[1]:
#					chan_epoch_indices_AF8[chan,epoch] = epoch
#	elif chan_epoch_indice_primary.any() == 2:
#		for chan in range(len(FT7_raw.ch_names)):
#			for epoch in range(50):
#				if AF8_power_45_70Hz[chan,epoch] >threshold_val[2]:
#					chan_epoch_indices_FT7[chan,epoch] = epoch
#	elif chan_epoch_indice_primary.any() == 3:
#		for chan in range(len(FT7_raw.ch_names)):
#			for epoch in range(50):
#				if AF8_power_45_70Hz[chan,epoch] >threshold_val[3]:
#					chan_epoch_indices_FT7[chan,epoch] = epoch
#print('thres_mara')
	
				
#Step 2a: Decomposing the power data array in to epochs by power



#Step 3: Locating the channels



#Step 3a: Decomposing the power data array into channels by epochs




#Step 4: Locating the contaminated epochs and finding the corresponding channel indices 
#for shuffling the contaminated data













