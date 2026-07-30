import re
import logging

import numpy as np

def bpr(ppt, di='+'):
    '''
    Apply a bipolar re-reference to the data. Subtract first adjacent contact from currenct contact.
    Order from depth to surface. Most superficial contact is removed.
    
    Parameters
    ----------
    ppt: dataclass
    ppt.exp.eeg: array (samples, channels)
        EEG time series
    ppt.exp.channels: array (electrodes, label)
        Channel names
    di: str
        Direction of the bipolar reference.
        + extracts contact +1, - extracts contact -1
    
    Returns
    ----------
    ppt: dataclass
    ppt.exp.eeg: array (samples, channels)
        Bipolar re-referenced EEG time series   
    ppt.exp.channels: array (electrodes, label)
        Channel names reduced (- last in shaft)
    '''

    shafts = set([ch.strip('0123456789') for ch in ppt.exp.channels])

    referenced = []
    channels = []

    for shaft in shafts:
        contact_idc = np.where(np.char.find(ppt.exp.channels, shaft) + 1)[0]
        referenced += [-np.diff(ppt.exp.eeg[:, contact_idc])]  # - to reverse order
        channels += [ppt.exp.channels[contact_idc[:-1]]]

    ppt.exp.eeg = np.hstack(referenced)
    ppt.exp.channels = np.concatenate(channels)

    return ppt