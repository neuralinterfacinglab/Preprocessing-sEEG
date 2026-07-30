# Builtin
import logging

# 3th party
import numpy as np
from scipy.stats import pointbiserialr
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def go(ppt):

    n_folds  = 10

    #Encode labels to 0 and 1
    x, y = ppt.exp.eeg, ppt.exp.labels
    le = LabelEncoder().fit(y)
    y = le.transform(y)

    #Initialize arrays
    ppt.icc = np.empty((n_folds, x.shape[1], x.shape[1]))
    ppt.pbc = np.empty((n_folds, x.shape[1], 2))
    icc_folds = np.zeros(n_folds)

    #Initialize folds
    folds = np.array_split(np.arange(y.size), n_folds)

    for i_fold, fold in enumerate(folds):

        #Inter-channel correlation
        iccfold = np.corrcoef(x[fold, :], rowvar=False)
        icc_folds[i_fold] = np.mean(np.abs(iccfold[np.triu_indices(iccfold.shape[0], k=1)]))
        ppt.icc[i_fold, :, :] =  iccfold

        #Task correlation
        ppt.pbc[i_fold, :, :] = np.apply_along_axis(pointbiserialr, 0, x[fold, :], y[fold]).T

    #Log information
    logging.info(f'{ppt.id} | Band: {ppt.band} | Prep: {ppt.prep} | ICC: {np.mean(icc_folds):.2f}')
    logging.info(f'{ppt.id} | Band: {ppt.band} | Prep: {ppt.prep} | PBC: {np.mean(np.abs(ppt.pbc[:,:,0])):.2f} | PBC Max: {np.mean(np.max(np.abs(ppt.pbc[:,:,0]), axis=1)):.2f}')    
                    
    return ppt