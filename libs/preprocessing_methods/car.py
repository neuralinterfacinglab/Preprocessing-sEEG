def car(ppt):
    '''
    Apply a common-average re-reference to the data
    
    Parameters
    ----------
    ppt: dataclass
    ppt.exp.eeg: array (samples, channels)  
        EEG time series
    
    Returns
    ----------
    ppt: dataclass
    ppt.exp.eeg: array (samples, channels)  
        CAR re-referenced EEG time series   
    '''

    ppt.exp.eeg = ppt.exp.eeg - ppt.exp.eeg.mean(axis=1, keepdims=True)

    return ppt