from tftb.processing import ShortTimeFourierTransform
from scipy.signal import hilbert
from scipy import angle, unwrap
import numpy as np
import matplotlib.pyplot as plt
from pyhht import EMD

def apply_EMD(raw):
    signal = raw.get_data()
    signal = np.resize(signal,(64,575))
#    s1 = np.sin(signal)
#    s2 = np.sin(signal)-1
#    s3 = np.sin(signal)+2
#    plt.plot(signal, s1, 'b', signal, s2, 'g', signal, s3, 'r'), plt.grid(),plt.show()
#    hs1 = hilbert(s1)
#    hs2 = hilbert(s2)
#    hs3 = hilbert(s3)
#    plt.plot(np.real(hs1), np.imag(hs1), 'b', np.real(hs2), np.imag(hs2), 'g',np.real(hs3), np.imag(hs3), 'r' ),plt.grid(),plt.show()






#    signal_stft = ShortTimeFourierTransform(signal)
#    signal_stft.run()
#    signal_stft.plot()
    hilbert_signal = hilbert(signal)
    omega_signal = unwrap(angle(hilbert_signal))
    f_inst_signal = np.diff(omega_signal)
#    time_samples = np.linspace(0,1,1000)
#    decomposer = EMD(signal)
#    imfs = decomposer.decompose()
    return omega_signal, f_inst_signal,  hilbert_signal  