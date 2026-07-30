import logging

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn import metrics

AXIS_0 = 0

def icc(ppt):
    '''
    Calculates interchannel correlations.

    '''
    
    # Inter-channel correlations
    df = pd.DataFrame(ppt.exp.eeg, columns = ppt.exp.channels.flatten().tolist())
    
    # Correlation matrix (can be used for stat analysis & plotting)
    ppt.icc = df.corr()
    
    # Average (absolute) score
    icc = np.average(ppt.icc.abs())  

    logging.info(f'{ppt.id} | Prep: {ppt.prep} | ICC: {icc:.2f}')
    
    return ppt

def pbc(ppt):
    ''' 
    Correlation between a continuous and a binary variable.
    NOTE: This function uses a shortcut formula but produces the same result as pearsonr. (as per scipy.stats documentation)
          i.e. more efficient
    
    '''
    label_map = {'Silence': 0, 'Speech': 1} if ppt.task.lower() == 'speech' else {'Rest': 0, 'Move': 1}
    
    binary_labels = np.array([label_map[label] for label in ppt.exp.labels])

    ppt.pbc, ppt.pbc_p = np.apply_along_axis(stats.pointbiserialr, AXIS_0, ppt.exp.eeg, binary_labels)

    logging.info(f'{ppt.id} | Prep: {ppt.prep} | PBC: {np.average(np.abs(ppt.pbc)):.2f} | PBC Significance: {(ppt.pbc_p<0.001).sum():d}/{ppt.exp.eeg.shape[1]}  | Absolute mean & n(p<0.001)/n')    
           
    return ppt

def esc(ppt):
    '''
    Calculates the correlation between eeg features and
    the log-mel spectrogram of a synchronized audio recording.
    Only applicable to the speech task.
    '''    
    corrs=[]

    for i in range(ppt.exp.eeg.shape[1]):
        c, p = stats.pearsonr(ppt.exp.eeg[:,i],np.mean(ppt.exp.mel_spec,axis=1))
        corrs.append(c)

    ppt.esc = corrs

    return ppt

def significant_channels(ppt, x, y):

    results = [stats.pearsonr(xi, y) for xi in x.T]

    ppt.significant_channels = np.vstack([(res.statistic, res.pvalue) for res in results])

    return ppt

def point_biserial_correlation(c, b):
    ''' Correlation between a continuous and a binary variable.
    --> Own implementation

    '''

    mask = np.where(b==0, True, False)
    
    m0, m1 = c[mask].mean(), c[~mask].mean()
    sn = c.std()
    
    n = b.size
    n0 = mask.sum()
    n1 = n - n0
    
    r_pb = (m1 - m0) / sn * np.sqrt(n1*n0 / n**2)

    logging.debug(m0, m1, sn, n, n0, n1, r_pb)
    logging.debug(f'R_pb: {r_pb:.2f} pearson: {np.corrcoef(c, b)[0, 1]:.2f}')
    pbs, p = stats.pointbiserialr(b, c)
    logging.debug(f'stats: {pbs:.3f}, p={p:.3f}, own: {r_pb:.3f}')

    return r_pb

def score_permutations(model, y, n_permutations=10000, per_fold=False):
    '''
    Shuffle the actual probabilities from the estimator (10000x) 
    and calculate roc_score again, make sure to check that the label
    is the correct index (may differ between folds)
    ''' 

    folds = model.folds
    y_hat = model.y_hat

    # Run some checks to be sure that the order is the same.
    #   Might be removed later
    assert all([y_hat[i].shape[0]==fold.shape[0] for i, fold in enumerate(folds)]), 'Fold size and y_hat size NOT equal.'
    assert all(np.concatenate([y[fold] for fold in folds]) == y),            'Folds are not in the same order as labels.'

    y_hat = np.hstack(y_hat)
    shuffled_scores = [metrics.roc_auc_score(y, np.random.permutation(y_hat)) \
                       for _ in np.arange(n_permutations)]

    model.s_shuffled = shuffled_scores

    return model

if __name__ == '__main__':

    # Random values
    c = np.random.random(1000)*100
    b = np.random.randint(0, 2, 1000)
    point_biserial_correlation(c, b)  # --> Should be 0
    

    # Square and sine wave with same frequency and phase
    c = np.sin(np.arange(0, 10, 0.1))
    b = np.where(c >= 0, 1, 0)
    point_biserial_correlation(c, b)  # --> Should be very high