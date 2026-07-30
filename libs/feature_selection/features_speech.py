# Builtin
import logging

# 3th Party
import numpy as np
import scipy

# Local
from libs.helpers.windowing import window, mode
from libs.helpers.extract_bands import get_bands
import libs.helpers.MelFilterBank as mel


def improve_labels(mel_spec):
    spec_avg = np.mean(mel_spec, axis=1)
    threshold = (np.max(spec_avg)+np.min(spec_avg))*0.45
    labels = np.where(spec_avg>threshold, 'Speech', 'Silence')
    return labels

def extract_mel_specs(audio, sr, wl=50, ws=10, n_filters=23):
    """Extract logarithmic mel-scaled spectrogram, traditionally used to compress audio spectrograms.
    
    Parameters
    ----------
    audio: array
        Audio time series
    sr: int
        Sampling rate of the audio
    wl: float
        Length of window (in milliseconds) in which spectrogram will be calculated
    ws: float
        Shift (in milliseconds) after which next window will be extracted
    n_filters: int
        Number of triangular filters in the mel filterbank
    
    Returns
    ----------
    spectrogram: array shape (n_windows, n_filters)
        Logarithmic mel-scaled spectrogram
    """

    wl *= 0.001
    ws *= 0.001
    numWindows=round((audio.shape[0]-wl*sr)/(ws*sr))
    win = scipy.hanning(np.floor(wl*sr + 1))[:-1]
    spectrogram = np.zeros((numWindows, int(np.floor(wl*sr / 2 + 1))),dtype='complex')
    for w in range(numWindows):
        startAudio = int(np.floor((w*ws)*sr))
        stopAudio = int(np.floor(startAudio+wl*sr))
        a = audio[startAudio:stopAudio]
        spec = np.fft.rfft(win*a)
        spectrogram[w,:] = spec
    mfb = mel.MelFilterBank(spectrogram.shape[1], n_filters, sr)
    spectrogram = np.abs(spectrogram)
    spectrogram = (mfb.toLogMels(spectrogram)).astype('float')
    return spectrogram

def get_windows(ppt, wl=1000, ws=200):
    w = window(ppt.exp.eeg, ppt.exp.ts, 
               wl, ws, ppt.exp.fs)
    ppt.exp.eeg = w.mean(axis=1)

    l = window(ppt.exp.labels, ppt.exp.ts,
              wl, ws, ppt.exp.fs)
    ppt.exp.labels = mode(l)

    return ppt

def go(ppt):

    if ppt.band == 'broadband':
        window_len = 50
        window_shift = 10   
    elif ppt.band == 'beta':    
        window_len = 250
        window_shift = 50
    else:
        logging.error(f'Not a valid frequency band: {ppt.band}')

    ppt.exp.eeg = get_bands(ppt.exp.eeg, ppt.exp.fs, ppt.band)
    ppt = get_windows(ppt, wl=window_len, ws=window_shift)
    
    ppt.exp.mel_spec = extract_mel_specs(ppt.exp.audio, ppt.exp.audio_fs, wl=window_len, ws=window_shift, n_filters=23)
    
    #Check for size differences, small amount (e.g., 1-2) is not an issue
    if ppt.exp.mel_spec.shape[0]!=ppt.exp.eeg.shape[0]:
        if np.abs(ppt.exp.eeg.shape[0]-ppt.exp.mel_spec.shape[0])>2:
            logging.warning(f'Possible Problem with EEG/Audio alignment for {ppt.id}, difference is {np.abs(ppt.exp.eeg.shape[0]-ppt.exp.mel_spec.shape[0])}')
        tLen = np.min([ppt.exp.mel_spec.shape[0],ppt.exp.eeg.shape[0]])
        ppt.exp.mel_spec = ppt.exp.mel_spec[:tLen,:]
        ppt.exp.eeg = ppt.exp.eeg[:tLen,:]
    
    ppt.exp.labels = improve_labels(ppt.exp.mel_spec)

    return ppt