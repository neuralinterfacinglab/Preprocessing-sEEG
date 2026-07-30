# Builtin
import logging
import re

# 3th party
import numpy as np

# Local
from libs.participant import Participant
from libs.preprocessing_methods.ica import icr
from libs.preprocessing_methods.bpr import bpr
from libs.preprocessing_methods.car import car
from libs.preprocessing_methods.cmr import cmr
from libs.preprocessing_methods.esr import esr
from libs.preprocessing_methods.lpr import lpr
from libs.preprocessing_methods.wm_only import wm_only
from libs.preprocessing_methods.wm_maj import wm_maj
from libs.preprocessing_methods.wm_prox import wm_prox
from libs.preprocessing_methods.noise import noise
from libs.preprocessing_methods.raw import raw

def get_func(method):
    if method == 'RR_CAR':  return car
    if method == 'RR_CMR':  return cmr
    if method == 'RR_ESR':  return esr
    if method == 'RR_LPR':  return lpr
    if method == 'RR_BPR':  return bpr
    if method == 'RR_ICR':  return icr
    if method == 'WM_ONLY': return wm_only
    if method == 'WM_MAJ':  return wm_maj
    if method == 'WM_PROX': return wm_prox
    if method == 'NOISE':   return noise
    if method == 'RAW':     return raw

def flag_disconnected_channels(ppt):
    pattern = '(?<![A-Za-z])[Ee][l\d]' # (E of e)l followed by a number
    return [i for i, channel in enumerate(ppt.exp.channels)\
              if re.search(pattern, channel)]

def flag_marker_channels(ppt):
    return [i for i, channel in enumerate(ppt.exp.channels)\
              if 'MKR' in channel]

def flag_ekg_channels(ppt):
    return [i for i, channel in enumerate(ppt.exp.channels)\
            if 'EKG' in channel]

def flag_flat_signal(ppt):
    return np.where(np.all(ppt.exp.eeg==ppt.exp.eeg[0], axis=0))[0]

def flag_pattern(ppt, patterns: list=[]):
    # NOTE: could be turned into regex for more versatility
    return np.hstack([i for i, channel in enumerate(ppt.exp.channels) \
                        for p in patterns \
                        if p in channel])
    
def remove_irrelevant_channels(ppt):
    '''
    - Marker channels
    - EKG channels
    - Disconnected channels
    - Flat signal
    '''
    flagged = np.hstack([flag_disconnected_channels(ppt),
                         flag_marker_channels(ppt),
                         flag_ekg_channels(ppt),
                         flag_flat_signal(ppt),
                         flag_pattern(ppt, ['+'])])
    flagged = np.unique(flagged).astype(int)
    
    ppt.exp.eeg = np.delete(ppt.exp.eeg, flagged, axis=1)
    ppt.exp.channels = np.delete(ppt.exp.channels, flagged)
    
    logging.info(f'{ppt.id} | Removed {len(flagged)} irrelevant channels.')
    
    return ppt

def find_segments(contact_nums):
    """
    Find all consecutive segments of contact numbers.
    Returns a list of segments, each segment is a list of numbers.
    """
    segments = []
    current_segment = [contact_nums[0]]
    # Find segments
    for i in range(1, len(contact_nums)):
        if contact_nums[i] == contact_nums[i - 1] + 1:
            current_segment.append(contact_nums[i])
        else:
            segments.append(current_segment)
            current_segment = [contact_nums[i]]
    # Append the last segment
    segments.append(current_segment)

    return segments

def compensate_incomplete_shafts(ppt):
    '''
    Compensate for shafts with missing contacts in somewhere in the middle,
    since this disproportionately affects LPR and BPR re-referencing methods
    '''
    # Function for later
    strip_letters = lambda s: s.strip('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
    # Extract unique shafts from original channels
    shafts = set([ch.strip('0123456789') for ch in ppt.exp.channels])
    n_shafts = len(shafts)
    n_channels = len(ppt.exp.channels)

    # Initialize list for excluded shafts and included channels
    excluded_shafts = []
    filtered_channels = []

    for shaft in shafts:
        # Get shaft-channel information
        contact_idc = np.where(np.char.find(ppt.exp.channels, shaft) + 1)[0]
        contacts_on_shaft = ppt.exp.channels[contact_idc]
        contact_nums = np.array([int(strip_letters(c)) for c in contacts_on_shaft])

        # Find all consecutive segments
        segments = find_segments(contact_nums)

        # Check for any valid segment of >= 3 channels, prioritizing the longest segment, the last if equal
        valid_segment = max((seg for seg in segments if len(seg) >= 3), key=lambda x: len(x), default=None)

        if valid_segment:
            # Add channels from the valid segment to filtered_channels
            for num in valid_segment:
                filtered_channels.append(f'{shaft}{num}')
        else:
            # Add shaft to excluded shafts
            excluded_shafts.append(shaft)

    # Filter channels and EEG data
    channel_mask = np.isin(ppt.exp.channels, filtered_channels)
    channel_mask_excluded = ~np.isin(ppt.exp.channels, filtered_channels)
    ppt.exp.excluded = ppt.exp.channels[channel_mask_excluded]
    ppt.exp.channels = np.array(filtered_channels)
    ppt.exp.eeg = ppt.exp.eeg[:,channel_mask]

    # Print information
    logging.info(f'{ppt.id} | Excluded shafts: {len(excluded_shafts)}/{n_shafts}')
    logging.info(f'{ppt.id} | Included channels: {len(ppt.exp.channels)}/{n_channels}')  

    return ppt 


def go(ppt: Participant, method: str) -> dict:   

    ppt = remove_irrelevant_channels(ppt)
    ppt = compensate_incomplete_shafts(ppt)
    ppt.prep = method

    method = get_func(method)

    ppt = method(ppt)
    logging.info(f'{ppt.id} | Prep: {ppt.prep} | Remaining channels: {len(ppt.exp.channels)}') 

    return ppt