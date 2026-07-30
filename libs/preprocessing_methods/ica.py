import re
import logging
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.feature_selection import chi2
from mne.filter import filter_data
from scipy.stats import chisquare

def icr(ppt, threshold=0.1):
    '''
    Apply an independent component analysis (ICA) to the data, discard 'broad' components and
    back-project to channel space, creating the independent component re-referenced (ICR).
    
    Parameters
    ----------
    ppt: dataclass
    ppt.exp.eeg: array (samples, channels)
        EEG time series
    ppt.exp.channels: array (electrodes, label)
        Channel names
    threshold: float
        Threshold for the pvalue of the chi2 test
    
    Returns
    ----------
    ppt: dataclass
    ppt.exp.eeg: array (samples, channels)
        ICA re-referenced EEG time series   
    ppt.discarded_comps: array(component index)
        Discarded component indices
    '''

    #high-pass filter of 1.5Hz
    eeg = filter_data(ppt.exp.eeg.T.astype('float64'), sfreq=ppt.exp.fs, l_freq=3, h_freq=None)
    ncomps = eeg.shape[0]
    ica = FastICA(n_components=ncomps, random_state=0, whiten='unit-variance', max_iter=250, tol=0.001)
    # Apply the ICA
    fitted = ica.fit_transform(eeg.T)
    # Get the mixing matrix
    matrix = ica.mixing_
    # Perform chi2 test on absolute of columns in matrix
    chi2_results = chisquare(np.abs(matrix))
    # Get broad components
    discard = np.where(chi2_results.pvalue > threshold)[0]

    # # Sort the matrix according to the chi2 pvalues
    # row_indices = np.repeat(np.arange(ncomps),ncomps).reshape(ncomps,ncomps)
    # col_indices = np.repeat(np.argsort(chi2_results.pvalue),ncomps).reshape(ncomps,ncomps)
    # matrix_sorted = matrix[row_indices, col_indices.T]
    # # Plot the matrix as sanity check
    # plt.imshow(matrix_sorted)
    # if len(discard) > 0:
    #     plt.axvline(np.where(np.sort(chi2_results.pvalue) > t)[0][0]-1, color='tab:orange', linewidth=3)
    # plt.colorbar()
    # plt.xlabel('Components (sorted)')
    # plt.ylabel('Channels')
    # # Save the figure for sanity check
    # plt.savefig(f'./figures/ICA/{ppt.exp.task}_{ppt.id}_{Path(__file__).stem}.png')
    # plt.close()

    # Remove the components above threshold
    fitted[:,discard] = 0
    # Do the inverse transform to get back to channel space
    restored = ica.inverse_transform(fitted)

    ppt.exp.eeg = restored
    ppt.discarded_comps = discard

    return ppt 
