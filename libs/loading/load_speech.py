import bisect
import logging
from pathlib import Path

import pyxdf
import numpy as np
from scipy.signal import decimate


def locate_pos(available_ts, target_ts):
    """Finds best fitting time point.
    
    Parameters
    ----------
    available_ts: array of floats 
        Time stamps of time series
    target_ts: float
        Time stamp that should be found in list of time stamps

    Returns
    ----------
    pos: integer 
        index of time stamp with smallest distance from target_ts
    """
    pos = bisect.bisect_right(available_ts, target_ts)
    if pos == 0:
        return 0
    if pos == len(available_ts):
        return len(available_ts)-1
    if abs(available_ts[pos]-target_ts) < abs(available_ts[pos-1]-target_ts):
        return pos
    else:
        return pos-1 


def load(path: Path) -> tuple[np.array, np.array,
                              np.array, float, np.array, 
                              np.array, int]:
    '''
    params:
        Path: Path() to .xdf file to load

    returns:
        data: 2d np.array       seeg data
        labels: 1d np.array     labels for each sample
        channels: 1d np.array   channels names given by amplifier
        fs: float               sampling frequency of amplifier
        ts: 1d np.array         time stamps for each sample
        audio: 1d np.array      audio data
        audio_fs: int           audio sampling frequency
    '''
    #Load xdf file
    streams = pyxdf.load_xdf(path,dejitter_timestamps=False)
    streamToPosMapping = {}
    for pos in range(0,len(streams[0])):
        stream = streams[0][pos]['info']['name']
        streamToPosMapping[stream[0]] = pos

    #Get sEEG data
    eeg = streams[0][streamToPosMapping['Micromed']]['time_series']
    # Get corresponding time stamps for each sample
    eeg_ts = streams[0][streamToPosMapping['Micromed']]['time_stamps'].astype('float')
    # Sampling rate
    eeg_sr = int(float(streams[0][streamToPosMapping['Micromed']]['info']['nominal_srate'][0]))
    
    # Some data sets are sampled with 2kHz, in that case we downsample
    if eeg_sr == 2048:
        eeg = decimate(eeg,2,axis=0)
        eeg_ts = eeg_ts[::2]
        eeg_sr = 2048/2

    # Get electrode info
    chNames = []
    for ch in streams[0][streamToPosMapping['Micromed']]['info']['desc'][0]['channels'][0]['channel']:
        chNames.append(ch['label'][0])

    #Load Audio
    audio = streams[0][streamToPosMapping['AudioCaptureWin']]['time_series']
    # Corresponding audio time stamps for each sample
    audio_ts = streams[0][streamToPosMapping['AudioCaptureWin']]['time_stamps'].astype('float')
    # Audio sampling rate
    audio_sr = int(streams[0][streamToPosMapping['AudioCaptureWin']]['info']['nominal_srate'][0]) 
    
    #Load Marker stream which contains the experiment timings
    markers = streams[0][streamToPosMapping['SingleWordsMarkerStream']]['time_series']
    marker_ts = streams[0][streamToPosMapping['SingleWordsMarkerStream']]['time_stamps'].astype('float')

    #Process Experiment timing
    i=0
    while markers[i][0]!='experimentStarted':
        i+=1
    # Find time stamp in eeg timestamps that is closest to the experiment start marker
    eeg_start= locate_pos(eeg_ts, marker_ts[i])
    # Find time stamp in audio timestamps that is closest to the experiment start marker
    audio_start = locate_pos(audio_ts, eeg_ts[eeg_start])
    while markers[i][0]!='experimentEnded':
        i+=1
    # Find time tamp in eeg timestamps that is closest to the experiment end marker
    eeg_end= locate_pos(eeg_ts, marker_ts[i])
    # Find time tamp in audio timestamps that is closest to the experiment end marker
    audio_end = locate_pos(audio_ts, eeg_ts[eeg_end])
    markers=markers[:i]
    marker_ts=marker_ts[:i]

    # Cut out only the audio and eeg during the experiment
    eeg = eeg[eeg_start:eeg_end,:]
    eeg_ts = eeg_ts[eeg_start:eeg_end]
    audio = audio[audio_start:audio_end,:]
    audio_ts=audio_ts[audio_start:audio_end]

    # Initialize corresponding labels for the time series
    words=['' for a in range(eeg.shape[0])]
    #Get only the starts for each word
    wordMask = [m[0].split(';')[0]=='start' for m in markers]
    wordStarts = marker_ts[wordMask]
    #Find the corresponding eeg time stamps
    wordStarts = np.array([locate_pos(eeg_ts, x) for x in wordStarts])
    #Extract only the word from the marker
    dispWords =  [m[0].split(';')[1] for m in markers if m[0].split(';')[0]=='start']
    # Same procedure for the word ends
    wordEndMask = [m[0].split(';')[0]=='end' for m in markers]
    wordEnds = marker_ts[wordEndMask]
    wordEnds = np.array([locate_pos(eeg_ts, x) for x in wordEnds])

    # Set the label vector with the words
    for i, start in enumerate(wordStarts):
        words[start:wordEnds[i]]=[dispWords[i].split('\r')[0] for rep in range(wordEnds[i]-start)]
    logging.info(f'{path.parent.name} | All aligned')
    
    #Add some white noise because the microphone thresholds the data
    noise = np.random.normal(0,0.0001, audio.shape[0])    
    audio = audio[:,0]+noise

    #Downsample audio to 16kHz
    audio_sr_target = 16000
    audio = decimate(audio,int(audio_sr / audio_sr_target))
    audio = np.int16(audio/np.max(np.abs(audio)) * 32767)
    audio_sr = audio_sr_target

    # Map to binary classes
    words = np.where(np.array(words)=='', 'Silence', 'Speech')

    data, labels, channels, fs, ts, audio, audio_fs = eeg, words, chNames, eeg_sr, eeg_ts, audio, audio_sr

    return data, labels, channels, fs, ts, audio, audio_fs