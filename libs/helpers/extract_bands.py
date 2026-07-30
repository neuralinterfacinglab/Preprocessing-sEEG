# Builtin

# 3th party
import numpy as np
import scipy

import mne
from mne.filter import filter_data
from mne.filter import notch_filter
from scipy import fftpack

mne.set_log_level('WARNING')

# Local

def get_bands(eeg, fs, band):
    '''
    Extract frequency-band envelope using the hilbert transform

    Parameters
    ----------
    data = 2d np.array [samples x channels]
    fs = int (sample rate: 1024 or 2048)
    bands = str (e.g 'beta', 'high_gamma')

    Returns
    ----------
        bands = 2d np.array [samples x (chs x bands)]
            data with filtered channels
    '''

    hilbert3 = lambda x: scipy.signal.hilbert(x, fftpack.next_fast_len(len(x)), 
                                              axis=0)[:len(x)]

    data = scipy.signal.detrend(eeg, axis=0)
    data = notch_filter(data.T.astype('float64'), fs, np.arange(50, 201, 50))

    filtered = []
    # for band in bands:
    if band == 'delta':      freqs = [.5, 4]
    if band == 'theta':      freqs = [4, 8]
    if band == 'alpha':      freqs = [8, 12]
    if band == 'beta':       freqs = [12, 30]
    if band == 'gamma':      freqs = [30, 55]
    if band == 'high_gamma': freqs = [55, 90]
    if band == 'broadband':  freqs = [70, 170]
    
    filtered += [filter_data(data,
                                sfreq=fs,
                                l_freq=freqs[0],
                                h_freq=freqs[1])]

    # Convert to power
    data = np.vstack(filtered)
    data = abs(hilbert3(data.T))

    return data