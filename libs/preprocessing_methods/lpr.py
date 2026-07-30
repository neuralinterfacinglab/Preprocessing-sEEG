import logging
import re

import numpy as np

def lpr(ppt, step_size=1):
    '''
    Apply a laplacian re-reference to the data
    
    Parameters
    ----------
    ppt: dataclass
    ppt.exp.eeg: array (samples, channels)
        EEG time series
    ppt.exp.channels: array (electrodes, label)
        Channel names
    step_size: int
        Size of the laplacian (amount of channels to include in average surrounding the channel)
    
    Returns
    ----------
    ppt: dataclass
    ppt.exp.eeg: array (samples, channels)
        Laplacian re-referenced EEG time series   
    '''

    shafts = set([ch.strip('0123456789') for ch in ppt.exp.channels])

    referenced = []
    for shaft in shafts:
        channel_idc = np.where(np.char.find(ppt.exp.channels, shaft) + 1)[0]  # +1 because returns 0 if true and -1 is false...

        # If at corner, act as bipolar. Replace else statement with np.zeros(ppt.exp[:, middle].shape) to switch to half laplacian
        referenced += [ppt.exp.eeg[:, middle] - (ppt.exp.eeg[:, left]  if left >=  channel_idc[0]  else ppt.exp.eeg[:, right] +
                                                 ppt.exp.eeg[:, right] if right <= channel_idc[-1] else ppt.exp.eeg[:, left]) / 2
                                                 for left, middle, right in zip(channel_idc - 1, 
                                                                                channel_idc,
                                                                                channel_idc + 1)]
                                                
    ppt.exp.eeg = np.vstack(referenced).T
    
    return ppt