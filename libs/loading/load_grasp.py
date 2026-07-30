from bisect import bisect_right

import numpy as np
from scipy.signal import decimate

from libs.helpers.read_xdf import xdf_to_dict

def locate_pos(ts, target_ts):
    pos = bisect_right(ts, target_ts)
    if pos == 0:
        return 0
    if pos == len(ts):
        return len(ts)-1
    if abs(ts[pos]-target_ts) < abs(ts[pos-1]-target_ts):
        return pos
    else:
        return pos-1    

def get_labels(eeg_ts, markers, marker_ts):
    # Create a label and trial numbers per timestamp

    # Find which markers correspond to the start and end of a trial
    trial_start_mask = [marker[0].split(';')[0]=='start' for marker in markers]
    trial_end_mask = [marker[0].split(';')[0]=='end' for marker in markers]

    # Find the indices corresponding to the start and end of the trial
    trial_idc_start = np.array([locate_pos(eeg_ts, trial) for trial in marker_ts[trial_start_mask]])
    trial_idc_end = np.array([locate_pos(eeg_ts, trial) for trial in marker_ts[trial_end_mask]])

    # Retrieve the corresponding labels per trial
    trial_labels = [marker[0].split(';')[1] for marker in markers if marker[0].split(';')[0] == 'start']

    # Map the label and trial number per index.
    trial_seq = ['Rest'] * eeg_ts.shape[0] # Trial labels sequential
    trial_nums = [0] * eeg_ts.shape[0]
    for i, idx_start in enumerate(trial_idc_start):
        trial_seq[idx_start:trial_idc_end[i]] = [trial_labels[i]] * (trial_idc_end[i]-idx_start)
        trial_nums[idx_start:trial_idc_end[i]] = [i] * (trial_idc_end[i]-idx_start)

    # Map to binary classes
    labels = np.where(np.array(trial_seq)=='Rest', 'Rest', 'Move')

    return labels

def get_exp_data(result):
    # Cuts out the relevant part of the data
    marker_idx_exp_start = result['GraspMarkerStream']['data'].index(['experimentStarted'])
    marker_idx_exp_end = result['GraspMarkerStream']['data'].index(['experimentEnded'])

    eeg_idx_exp_start = locate_pos(result['Micromed']['ts'], 
                                result['GraspMarkerStream']['ts'][marker_idx_exp_start])
    eeg_idx_exp_end = locate_pos(result['Micromed']['ts'],
                                result['GraspMarkerStream']['ts'][marker_idx_exp_end])

    eeg = result['Micromed']['data'][eeg_idx_exp_start:eeg_idx_exp_end, :]
    eeg_ts = result['Micromed']['ts'][eeg_idx_exp_start:eeg_idx_exp_end]

    marker = result['GraspMarkerStream']['data'][marker_idx_exp_start:marker_idx_exp_end]
    marker_ts = result['GraspMarkerStream']['ts'][marker_idx_exp_start:marker_idx_exp_end]

    return eeg, eeg_ts, marker, marker_ts

def load(path):
    '''
    Entry point of file.
    '''
    data, _ = xdf_to_dict(path)
    eeg, eeg_ts, markers, marker_ts = get_exp_data(data)
    labels = get_labels(eeg_ts, markers, marker_ts)
    channels = np.array(data['Micromed']['channel_names'])
    fs = data['Micromed']['fs']

    return eeg, labels, channels, fs, eeg_ts