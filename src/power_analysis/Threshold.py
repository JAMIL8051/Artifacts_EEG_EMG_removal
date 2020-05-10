import sim_data as sd


#Function to find band power of the simulated data
def final_band_power_sim(raw,channel,N,fs,tmin=None, tmax=None,epoch_time=2):
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
    
    if (tmin==None and tmax == None):
        n_epochs = 50
    else:
        n_epochs = int((tmax-tmin)/epoch_time)
    
    data = raw.get_data()
    data = data.transpose(1,2,0).reshape(len(raw.ch_names),-1)
    data = data/1e-06 # Unit conversion to microvolts from volts
     
    n_channels = len(raw.ch_names)
    epoch_power_45_70Hz = np.zeros((n_channels,n_epochs,1),dtype = 'float')
    
    for chan in range(n_channels):
        for epoch in range(n_epochs):
            epoch_data = data[chan, ((N*epoch)+1):(N*(epoch+1)+1)] #Formation of 2s epoch data with N samples
            epoch_data = np.fft.fft(epoch_data, N)
            
            #calculation of k in the formula
            k_lower = int(45*(N/fs)) # Forcing these to integers otherwise index error problem 
            # will pop up 
            k_upper = int(70*(N/fs))
            
            temp = np.zeros((50),dtype='float')
            
            for k in range(k_lower, k_upper):
                #val = (np.abs(epoch_data[k_lower])) ** 2
                val = (np.abs(epoch_data[k_lower])) ** 2 + (np.abs(epoch_data[N - k_lower])) ** 2
                temp[(k-90)] = val
            
            power_45_70Hz = (1/N**2)*sum(temp)
            epoch_power_45_70Hz[chan,epoch] = power_45_70Hz
            
    return epoch_power_45_70Hz


#Function for getting the threshold by simulatin the real data provided by the user
def sim_threshold(raw):
    events = mne.find_events(raw, shortest_event=0, stim_channel='Status')

    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=True,
                       exclude='bads')

    event_id ={'2-back_spatial': 20, '3-back_spatial': 30, '1-back_nonspatial': 40}
    tmin, tmax = -1., 6.5
    n_back_epo = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                            baseline=None, preload=True)
    orders = np.arange(5, 28, 1)  # The orders to search
    mse_res, rmse_res, r2_res, aic_res, loglik_res = sd.search_for_best_order(n_back_epo,
                                                                          orders=orders, 
                                                                          plot=True)
    best_order = orders[np.asarray(mse_res).argmin()]
    sim_data_raw_object = sd.make_sim_data(n_back_epo, baseline=(None, -0.1), order=17)

