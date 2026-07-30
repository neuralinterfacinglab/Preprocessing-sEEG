import numpy as np

def esr(ppt):
    '''
    Apply an electrode-shaft re-reference to the data
    
    Parameters
    ----------
    ppt: dataclass
    ppt.exp.eeg: array (samples, channels)      
        EEG time series
    ppt.exp.channels: array (electrodes, label) 
        Channel names
    
    Returns
    ----------
    ppt: dataclass
    ppt.exp.eeg: array (samples, channels)      
        ESR re-referenced EEG time series  
    '''    

    data_ESR = np.zeros((ppt.exp.eeg.shape[0], ppt.exp.eeg.shape[1]))

    # Get shaft information
    shafts = {}
    for i, chan in enumerate(ppt.exp.channels):
        if chan[0].rstrip('0123456789') not in shafts:
            shafts[chan[0].rstrip('0123456789')] = {'start': i, 'size': 1}
        else:
            shafts[chan[0].rstrip('0123456789')]['size'] += 1

    # Get average signal per shaft
    for shaft in shafts:
        shafts[shaft]['average'] = np.average(ppt.exp.eeg[:, shafts[shaft]['start']:(shafts[shaft]['start'] + shafts[shaft]['size'])], axis=1)

    # Subtract the shaft average from each respective channel   
    for i in range(ppt.exp.eeg.shape[1]):
        data_ESR[:, i] = ppt.exp.eeg[:, i] - shafts[ppt.exp.channels[i][0].rstrip('0123456789')]['average']
    
    ppt.exp.eeg = data_ESR

    return ppt