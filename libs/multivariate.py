# Builtin
import logging
from time import time
from dataclasses import dataclass

# 3th party
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics

debug = False

@dataclass
class LearnedClassifier:
    clf:   Pipeline   # This is the classifier trained on the last fold.
    y_hat: np.array
    s_tr:  np.array  # Could separate scores out.
    s_te:  np.array
    folds: np.array

def score_permutations(model, y, n_permutations=1000, per_fold=False):
    '''
    Shuffle the actual probabilities from the estimator 
    and calculate roc_score again, make sure to check that the label
    is the correct index (may differ between folds)
    ''' 

    folds = model.folds
    y_hat = model.y_hat

    # Run some checks to be sure that the order is the same.
    assert all([y_hat[i].shape[0]==fold.shape[0] for i, fold in enumerate(folds)]), 'Fold size and y_hat size NOT equal.'
    assert all(np.concatenate([y[fold] for fold in folds]) == y),            'Folds are not in the same order as labels.'

    y_hat = np.hstack(y_hat)
    shuffled_scores = [metrics.roc_auc_score(y, np.random.permutation(y_hat)) \
                       for _ in np.arange(n_permutations)]

    model.s_shuffled = shuffled_scores

    return model


def score(y, y_hat):
    return np.array([
        metrics.roc_auc_score(y, y_hat[:, 1])
        ])


def get_learner(method):

    if method == 'allchs':
        learner = Pipeline([('scaler', StandardScaler()),
                            ('kbest', SelectKBest(score_func=f_regression, k='all')),
                            ('model', LinearDiscriminantAnalysis())])

    elif method == 'selchs':
        learner = Pipeline([('scaler', StandardScaler()),
                            ('kbest', SelectKBest(score_func=f_regression, k=10)),
                            ('model', LinearDiscriminantAnalysis())])

    elif method == 'selpca':
        learner = Pipeline([('scaler', StandardScaler()),
                            ('pca', PCA(n_components=10)),
                            ('model', LinearDiscriminantAnalysis())])

    return learner


def train_test_split(x, y, fold):
    
    test_x, test_y = x[fold, :], y[fold]
    train_x, train_y = np.delete(x, fold, axis=0), np.delete(y, fold, axis=0)

    return train_x, train_y, test_x, test_y


def go(ppt, method):

    #Parameters
    n_folds  = 10
    n_scores = 1
    n_perms = 100

    #Label encoder
    x, y = ppt.exp.eeg, ppt.exp.labels
    le = LabelEncoder().fit(y)
    y = le.transform(y)

    #Initialize arrays
    y_hat_test = []
    scores_train = np.empty(n_folds)
    scores_test  = np.empty(n_folds)

    #Intialize folds
    folds = np.array_split(np.arange(y.size), n_folds)

    for i_fold, fold in enumerate(folds):
        # t = time()

        #Split data into folds
        train_x, train_y, test_x, test_y = train_test_split(x, y, fold)

        #Fit the data
        learner = get_learner(method)
        learner.fit(train_x, train_y)

        #Predict the data
        yh_train = learner.predict_proba(train_x)
        yh_test  = learner.predict_proba(test_x)
    
        #Calculate performance
        scores_train[i_fold] = score(train_y, yh_train)
        scores_test[i_fold] = score(test_y, yh_test)

        y_hat_test += [yh_test[:, 1]] # Cant stack, not all folds are equal size

        # logging.info(f'{ppt.id} | Fold {i_fold} fit in {time()-t:.3f}s')

    #Collect results in structure
    res = LearnedClassifier(learner, y_hat_test, scores_train, scores_test, folds)

    #Optional: permutate the data to calculate chance level
    res = score_permutations(res, ppt.exp.labels, n_permutations=n_perms)

    #Save the data into participant structure
    if method == 'allchs':
        ppt.allchs = res
    elif method == 'selchs':
        ppt.selchs = res
    elif method == 'selpca':
        ppt.selpca = res

    #Log information
    logging.info(f'{ppt.id} | Band: {ppt.band} | Prep: {ppt.prep} | Method: {method} | AUC: {res.s_te.mean():.2f} +- {res.s_te.std():.2f}')
                
    return ppt