# Builtin
import logging
from time import time
from dataclasses import dataclass

# 3th party
import numpy as np
from scipy.stats import pearsonr, pointbiserialr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn import metrics

from libs.metrics import significant_channels

TRAIN, TEST = 0, 1
FEATURE_SELECTION = 1

ALPHA = 0.05
CORRECTION = 'bonferonni'

# # Local
# GRID_PARAMS = {'linear':    {'model__solver':            ['svd']},

#                'nonlinear': {'model__max_depth':         [5, 10],          # Maximum depth per tree
#                              'model__learning_rate':     [0.3, 0.5, 0.8],      # Steps size per iteration
#                              'model__n_estimators':      [10, 100, 200],  # number of trees
#                              'model__colsample_bytree':  [1],                  # Fraction of columns per sample
#                              'model__subsample':         [0.8, 1],             # Fraction of observations sampled per tree
#                             #  'model__alpha':             [],                 # L1 regularization,
#                              'model__lambda':            [0.1, 0.5, 1],        # L2 regularization,
#                }}

debug = False

@dataclass
class LearnedClassifier:
    clf:   Pipeline   # This is the classifier trained on the last fold.
    y_hat: np.array
    s_tr:  np.array  # Could separate scores out.
    s_te:  np.array
    folds: np.array

def score(y, y_hat):
    return np.array([
        metrics.roc_auc_score(y, y_hat[:, 1])
        ])

def by_pearsonr(X, y):
    y = y[:, np.newaxis] if y.ndim == 1 else y
    xy = np.hstack((X, y))  # Stack for convenience
    return np.abs(np.corrcoef(xy, rowvar=False)[:-1, -1])

def by_sigcorr(X, y):
    # return np.array([pearsonr(ch, y) for ch in X.T])
    return np.array([pearsonr(ch, y).p_value for ch in X.T])

def get_learner(train_x, train_y, method):

    if method == 'pca':
        learner = Pipeline([('scaler', StandardScaler()),
                            ('pca', PCA(n_components=10)),
                            ('model', LinearDiscriminantAnalysis())])
    
    elif method == '10_corr':
        learner = Pipeline([('scaler', StandardScaler()),
                            ('kbest', SelectKBest(score_func=f_regression, k=10)),
                            ('model', LinearDiscriminantAnalysis())])

    elif method == 'sig_chs':
        learner = Pipeline([('scaler', StandardScaler()),
                            ('kbest', SelectKBest(score_func=f_regression, k='all')),
                            ('model', LinearDiscriminantAnalysis())])

        learner.fit(train_x, train_y)

        # get the significant features
        p_values = learner.named_steps['kbest'].pvalues_
        n_sig_channels = np.where(p_values * (train_x.shape[1] if CORRECTION=='bonferonni' else 1) < ALPHA)[0].size

        # update n k-values accordingly
        learner.named_steps['kbest'].k = n_sig_channels

    return learner

# def get_folds_stratified(y, n_folds):
#     # y.shape = [samples x classes] or usually [n x 1]

#     return StratifiedKFold(n_folds, shuffle=False).split(np.arange(y.shape[0]), y)

def cv_signal_metrics(ppt, i_fold, train_x, train_y, test_x, test_y):

    ppt.cv_icc_train[i_fold, :, :] = np.corrcoef(train_x, rowvar=False)
    ppt.cv_icc_test[i_fold, :, :] =  np.corrcoef(test_x, rowvar=False)

    ppt.cv_pbc_train[i_fold, :, :] = np.apply_along_axis(pointbiserialr, 0, train_x, train_y).T
    ppt.cv_pbc_test[i_fold, :, :] = np.apply_along_axis(pointbiserialr, 0, train_x, train_y).T


def train_test_split(x, y, fold):
    
    test_x, test_y = x[fold, :], y[fold]
    train_x, train_y = np.delete(x, fold, axis=0), np.delete(y, fold, axis=0)

    return train_x, train_y, test_x, test_y

def go(ppt, method):

    n_folds  = 10
    n_scores = 1

    x, y = ppt.exp.eeg, ppt.exp.labels
    le = LabelEncoder().fit(y)
    y = le.transform(y)

    ppt.cv_icc_train = np.empty((n_folds, x.shape[1], x.shape[1]))
    ppt.cv_icc_test = np.empty((n_folds, x.shape[1], x.shape[1]))

    ppt.cv_pbc_train = np.empty((n_folds, x.shape[1], 2))
    ppt.cv_pbc_test = np.empty((n_folds, x.shape[1], 2))

    selected_channels = []
    y_hat_test = []
    scores_train = np.empty((n_folds, n_scores))
    scores_test  = np.empty((n_folds, n_scores))

    folds = np.array_split(np.arange(y.size), n_folds)

    logging.info(f'{ppt.id} | Start fitting model')
    for i_fold, fold in enumerate(folds):
        t = time()

        train_x, train_y, test_x, test_y = train_test_split(x, y, fold)

        cv_signal_metrics(ppt, i_fold, train_x, train_y, test_x, test_y)

        # fit_predict(train_x, train_y, test_x, test_y)
        learner = get_learner(train_x, train_y, method)
        learner.fit(train_x, train_y)

        yh_train = learner.predict_proba(train_x)
        yh_test  = learner.predict_proba(test_x)
    
        scores_train[i_fold, :] = score(train_y, yh_train)
        scores_test[i_fold, :] = score(test_y, yh_test)

        y_hat_test += [yh_test[:, 1]] # Cant stack, not all folds are equal size

        if method != 'pca':
            scores = np.vstack([np.arange(train_x.shape[1]), learner.named_steps['kbest'].scores_, learner.named_steps['kbest'].pvalues_]).T
            selected_channels += [scores[learner[FEATURE_SELECTION].get_support()]]

        logging.info(f'{ppt.id} | Fold {i_fold} fit in {time()-t:.3f}s')

    ppt.lin = LearnedClassifier(learner, y_hat_test, scores_train, scores_test, folds)
    ppt.selected_channels = selected_channels

    logging.info(f'{ppt.id} | Prep: {ppt.prep} | AUC: {ppt.lin.s_te.mean():.3f} +- {ppt.lin.s_te.std():.3f}')
                
    return ppt