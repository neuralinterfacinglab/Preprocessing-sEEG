import logging

import numpy as np

def wm_maj(ppt):
    '''
    Exclude electrodes with a Proximal Tissue Density (PTD) between -1 and 0.
    This means including only electrodes of which the surrounding voxels are mostly gray (or subcortical) matter,
    excluding electrodes where the majority if its surrouding voxels are white matter.
    
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
            logging.warning(f'kh{ppt.kh_id:03d} | WM_MAJ | Channel {ch} not in ppt.PTD.keys(). Skipping this channel.')
            continue
            
        if val < 0 or np.isnan(val):
            chs_to_remove += [i]

    ppt.exp.eeg = np.delete(ppt.exp.eeg, chs_to_remove, axis=1)
    ppt.exp.channels = np.delete(ppt.exp.channels, chs_to_remove)

    return ppt