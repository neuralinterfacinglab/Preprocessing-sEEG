import logging

import numpy as np

def wm_prox(ppt):
    '''
    Exclude electrodes with a Proximal Tissue Density (PTD) of anything below 1.
    This means including only electrodes fully in and surrounded by gray (or subcortical) matter,
    excluding anything with white matter in the proximity.
    
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
            logging.warning(f'kh{ppt.kh_id:03d} | WM_PROX | Channel {ch} not in ppt.PTD.keys(). Skipping this channel.')
            continue
            
        if val < 1 or np.isnan(val):
            chs_to_remove += [i]

    ppt.exp.eeg = np.delete(ppt.exp.eeg, chs_to_remove, axis=1)
    ppt.exp.channels = np.delete(ppt.exp.channels, chs_to_remove)

    return ppt