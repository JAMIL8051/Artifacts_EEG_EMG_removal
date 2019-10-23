import mne

BIOSEMI_MACHINE_NAME = 'biosemi'
DEFAULT_MONTAGE_MACHINE ='biosemi64'
DEFAULT_EOG = ['EXG1','EXG2','EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']


def load_montage(channel_names,
                 channel_file_path, 
                 kind = DEFAULT_MONTAGE_MACHINE,
                 unit ='m',
                 transform = False):
    montage = mne.channels.read_montage(kind, channel_names, channel_file_path, unit, transform)
    return montage

def load_biosemi_montage(biosemi_channel_count = 64):
    montage = load_montage(kind = BIOSEMI_MACHINE_NAME + str(BIOSEMI_MACHINE_NAME))
    return montage

def load_raw_data(filepath, montage, eog):
    raw = mne.io.read_raw_edf(filepath, montage = montage, eog)
    return raw