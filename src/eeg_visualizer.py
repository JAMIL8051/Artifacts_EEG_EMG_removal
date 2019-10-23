
def print_ICA(ica, filtered_raw):
##Plotting of the ICA components
    ica.plot_components()#Some potential bad guys can be spotted
#
##First component 0 have a close look
    ica.plot_properties(filtered_raw, picks=0)
#
##Data was fitlered so less informative sprectrum plot. Hence we change:
    ica.plot_properties(filtered_raw, picks=0,psd_args={'fmax': 250.})

#Multiple different components at once
    ica.plot_properties(filtered_raw, picks=[1,2,3],psd_args={'fmax':500.})

def print_average_eog(average_eog):
    print('We found %i EOG events' % average_eog.nave)
    joint_kwargs =dict(ts_args=dict(time_unit='s'))
    average_eog.plot_joint(**joint_kwargs)