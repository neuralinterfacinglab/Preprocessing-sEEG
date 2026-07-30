# Builtin
import logging

# 3th Party
import numpy as np

# Local
from libs.helpers.windowing import window, mode
from libs.helpers.extract_bands import get_bands

def spectral_filters(ppt, bands: list):
    ppt.exp.eeg = get_bands(ppt.exp.eeg, ppt.exp.fs, bands)

    return ppt

def get_windows(ppt, wl=1000, ws=200):
    w = window(ppt.exp.eeg, ppt.exp.ts, 
               wl, ws, ppt.exp.fs)
    ppt.exp.eeg = w.mean(axis=1)

    l = window(ppt.exp.labels, ppt.exp.ts,
              wl, ws, ppt.exp.fs)
    ppt.exp.labels = mode(l)

    return ppt

def go(ppt):

    if ppt.band == 'beta':
        window_len = 500 #1000
        window_shift = 100 #200  
    elif ppt.band == 'broadband':    
        window_len = 100
        window_shift = 20
    else:
        logging.error(f'Not a valid frequency band: {ppt.band}')

    ppt = spectral_filters(ppt, ppt.band)
    ppt = get_windows(ppt, wl=window_len, ws=window_shift)

    return ppt