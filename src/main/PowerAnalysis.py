import numpy as np
import mne


#def idntifyArtifactsForChannel(raw,channels):
	
#	power45_70Hz = bandPower(raw, channels)
#	data = raw.get_data()

	
#Global function for identification of EMG artifacts by 
#power analysis in the 45-70Hz freuquency band.

def identifyArtifacts(raw):
	#Duplicating the raw object: Util function??
	primaryPower45_70Hz, primaryData = bandPower(raw, channels = ['AF7','AF8','FT7','FT8'])
	af7Power45_70Hz, af7Data = bandPower(raw, channels =['Fp1', 'AF3', 'F3', 'F5', 'F7'])
	af8Power45_70Hz, af8Data = bandPower(raw, channels = ['Fp2', 'AF4', 'F4', 'F6', 'F8'])
	ft7Power45_70Hz, ft7Data = bandPower(raw, channels = ['F7', 'F5', 'FC5', 'C5', 'T7'])
	ft8Power45_70Hz, ft8Data = bandPower(raw, channels = ['F8', 'F6', 'FC6', 'C6', 'T8'])
	fzPower45_70Hz, fzData = bandPower(raw, channels = ['Fz'])
	

	#Step 1 :Finding the threshold val without simulated data.

	#thresholdValPaper = primaryPower45_70Hz.mean(axis =1)+1*primaryPower45_70Hz.std(axis =1) 
	threshold_val = priprimaryPower45_70Hz.mean(axis = 0)

	#Allocating memory for the channel and epoch indices array for the primary channels data
	chan_epoch_indice_primary = np.zeros((len(primary_raw.ch_names),100),dtype='int')

	for chan in range(len(primary_raw.ch_names)):
		for epoch in range(100):
			if primary_power_45_70Hz[chan,epoch] > threshold_val[chan]:
				if epoch == 0:
					chan_epoch_indice_primary[chan,epoch] = -1 #Setting -1 for the first epoch
				else:
					chan_epoch_indice_primary[chan,epoch] = epoch

	#Storing the primary contaminated data obtained from channel and epoch indices                
	primary_contaminated_data = np.zeros((len(primary_picks),100,1024),dtype='float')

	for chan in range(len(primary_picks)):
		for epoch in range(100):
			if epoch == 0:
				if chan_epoch_indice_primary[chan,epoch] == -1:
					primary_contaminated_data[chan,epoch,:] = primary_data[chan, (1024*epoch)+1:(1024*(epoch+1))+1]
			elif chan_epoch_indice_primary[chan,epoch]!= 0:
				primary_contaminated_data[chan,epoch,:] = primary_data[chan, (1024*epoch)+1:(1024*(epoch+1))+1]

	# Getting the channel, epoch and time point indices of the non-zero contaminated data
	channels, epochs,times = np.nonzero(primary_contaminated_data != 0)
	
	# Setting the no. of channels and no. of time points 
	n_channels = len(primary_raw.ch_names)
	n_time_points = int(len(times)/n_channels)
	dataWithArtifactsDetected = primary_contaminated_data[primary_contaminated_data!=0].reshape(n_time_points,n_channels)

	ch_names = primary_raw.ch_names
	n_channels = len(primary_raw.ch_names)
	sampling_rate = raw.info['sfreq']
	info = mne.create_info(ch_names=ch_names, sfreq=sampling_rate, ch_types=['eeg']*n_channels)
	dataWithArtifactsDetectedRaw = mne.io.RawArray(dataWithArtifacts, info)

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





# Function for Power_analysis
def finalBandPower(raw,channel,N,fs,tmin=None, tmax=None,epoch_time=2):
	"""
		raw = Mne.raw object
		picks = no. of channels to include, type must be string
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
	epoch_power_45_70Hz = np.zeros((n_channels,n_epochs,1),dtype = 'float')
	
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
			temp = np.zeros((50),dtype='float')
			for k in range(k_lower, k_upper):
				val = (np.abs(epoch_data[k_lower])) ** 2 + (np.abs(epoch_data[N - k_lower])) ** 2
				temp[k-90] = val
			
			power_45_70Hz = (1/N**2)*sum(temp)
			epoch_power_45_70Hz[chan,epoch] = power_45_70Hz
			
	return epoch_power_45_70Hz, data*1e-06


#Pore description likbo
	def bandPower(raw,channels):
		filteredRaw = raw.copy().pick(picks=channels)
		return finalBandPower(filteredRaw,channel = channels, N=1024,fs=512) 




