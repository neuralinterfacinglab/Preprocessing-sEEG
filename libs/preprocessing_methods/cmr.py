import numpy as np

def cmr(ppt):
    '''
    Apply a common-median re-reference to the data
    
    Parameters
    ----------
    ppt: dataclass
    ppt.exp.eeg: array (samples, channels)  
        EEG time series
    
    Returns
    ----------
    ppt: dataclass
    ppt.exp.eeg: array (samples, channels)  
        CMR re-referenced EEG time series   
    '''


    ppt.exp.eeg = ppt.exp.eeg - np.median(ppt.exp.eeg, axis=1, keepdims=True)

    return ppt