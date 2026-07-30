import logging

import numpy as np

def wm_only(ppt):
    '''
    Exclude electrodes with a Proximal Tissue Density (PTD) of -1.
    This means excluding only electrodes fully in and surrounded by white matter,
    including anything with gray (or subcortical) matter in the proximity.
    
    Parameters
    ----------
    ppt: dataclass
    ppt.exp.eeg: array (samples, channels)
        EEG time series
    ppt.exp.channels: array (electrodes, label)
        Channel names
    ppt.PTD: dict (electrode label: PTD value)
        Dictionary linking electrode with PTD value

    Returns
    ----------
    ppt: dataclass
    ppt.exp.eeg: array (samples, channels)
        Reduced EEG time series   
    ppt.exp.channels: array (electrodes, label)
        Reduced channel names
    
    '''
    chs_to_remove = []
    for i, ch in enumerate(ppt.exp.channels):
        val  = ppt.PTD.get(ch, 'NoValue')

        if val == 'NoValue':
            logging.warning(f'kh{ppt.kh_id:03d} | WM_ONLY | Channel {ch} not in ppt.PTD.keys(). Skipping this channel.')
            continue
            
        if val < -1 or np.isnan(val):
            chs_to_remove += [i]

    ppt.exp.eeg = np.delete(ppt.exp.eeg, chs_to_remove, axis=1)
    ppt.exp.channels = np.delete(ppt.exp.channels, chs_to_remove)

    return ppt