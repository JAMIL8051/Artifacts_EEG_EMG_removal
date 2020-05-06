import matplotlib.pyplot as plt
import numpy as np
import mne


#Plotting the detected EOG events 
def print_average_eog(average_eog):
    print('We found %i EOG events' % average_eog.nave)
    joint_kwargs = dict(ts_args=dict(time_unit='s'))
    average_eog.plot_joint(**joint_kwargs)

#Plotting the results of application of ICA
def print_ICA(ica, filtered_raw, eog_average, eog_epochs, eog_inds, scores, raw_copy):
    # Plotting of the ICA components
    ica.plot_components()  # Some potential bad guys can be spotted

# First component 0 have a close look
    #ica.plot_properties(filtered_raw, picks=0)

# Data was fitlered so less informative sprectrum plot. Hence we change:
    #ica.plot_properties(filtered_raw, picks=0,psd_args={'fmax': 250.})

# Multiple different components at once
    ica.plot_properties(filtered_raw, picks=[1, 2, 3], psd_args={'fmax': 500.})

# look at r scores of components and can see only one component highly correlated
# and this one got detected by our correlation analysis(red)
    ica.plot_scores(scores, exclude=eog_inds)

# Looking at source time courses
    ica.plot_sources(eog_average, exclude=eog_inds)

# Properties of component with data epoched with respect to EOG events and use
# of little smoothing along trials axis in epochs image
    ica.plot_properties(eog_epochs, picks=eog_inds, psd_args={
                        'fmax': 35.}, image_args={'sigma': 1.})
    print(ica.labels_)

# Modifiying singnals if component is removed from data
    ica.plot_overlay(eog_average, exclude=eog_inds, show=False)

# Red is before, black after. We remove quite a bit

# Registering this component as bad to be removed using "ica.exclude"
    ica.exclude.extend(eog_inds)

# from this ICA will reject this component even if no exclude parameter is passed,
# and this info be stored to disk on saving

#reading and writing
#   ica.save('my-ica.bdf')
#   ica = read_ica('my-ica.bdf')
    raw_copy.plot()  # Checking result





def plot_EOG_artifacts(epochs, data):
    plt. plot(1e3*epochs.times, np.resize(data,
                                          (len(epochs.times), len(data))))
    plt.xlabel('Time (ms)')
    plt.ylabel('EOG (muV)')
    plt.show()


def get_projections(projs, raw, eog_projs):
    print(projs)
    mne.viz.plot_projs_topomap(eog_projs, info=raw.info)


def plot_demonstration_SSP_cleaning(epochs_no_proj, epochs_proj, evoked):
    epochs_no_proj.average().plot(spatial_colors= True, time_unit='s')
    epochs_proj.average().plot(spatial_colors= True, time_unit='s')

# Plotting delayed SP mode and setting time  instances from 50ms to 200ms in a step of 10ms
    times = np.arange(0.05, 0.20, 0.01)
    fig = evoked.plot_topomap(times, proj='interactive', time_unit='s')
    return fig
