import logging

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

def prctile(x, p):
    p = np.asarray(p, dtype=float)
    n = len(x)
    p = (p - 50) * n / (n - 1) + 50
    p = np.clip(p, 0, 100)
    return np.percentile(x, p)

def heuristic_bad_chs(ppt):
    """
    Flags channels with poor signal quality based on several heuristic criteria.
    Based on the pipeline by Shi et al. (2024): https://doi.org/10.1093/braincomms/fcae165 

    Criteria
    --------
    - >50% NaNs
    - >50% zeros
    - Excessively large amplitudes
    - Rare extreme outliers
    - Excessive 50 Hz line noise
    - Standard deviation much larger than the other channels

    Parameters
    ----------
    ppt : Participant instance

    Returns
    -------
    ppt : Participant instance
    """

    data = ppt.exp.eeg
    fs = ppt.exp.fs

    # Parameters
    tile = 99
    mult = 10
    num_above = 1
    abs_thresh = 5e3

    # 50 Hz line noise threshold
    percent_50_hz = 0.7

    # Standard deviation threshold
    mult_std = 10

    nchs = data.shape[1]

    bad = []
    high_ch = []
    nan_ch = []
    zero_ch = []
    high_var_ch = []
    noisy_ch = []

    all_std = np.nanstd(data, axis=0)
    all_bl = np.nanmedian(data, axis=0)

    for ich in range(nchs):

        eeg = data[:, ich]

        # Too many NaNs
        if np.sum(np.isnan(eeg)) > 0.5 * len(eeg):
            bad.append(ich)
            nan_ch.append(ich)
            continue

        # Too many zeros
        if np.sum(eeg == 0) > 0.5 * len(eeg):
            bad.append(ich)
            zero_ch.append(ich)
            continue

        # Too many very large amplitudes
        if np.sum(np.abs(eeg - all_bl[ich]) > abs_thresh) > 10:
            bad.append(ich)
            high_ch.append(ich)
            continue

        # Rare extreme outliers
        pct = prctile(eeg, [100 - tile, tile])
        thresh = [
            all_bl[ich] - mult * (all_bl[ich] - pct[0]),
            all_bl[ich] + mult * (pct[1] - all_bl[ich]),
        ]

        sum_outside = np.sum(eeg > thresh[1]) + np.sum(eeg < thresh[0])

        if sum_outside >= num_above:
            bad.append(ich)
            high_var_ch.append(ich)
            continue

        # Excessive 50 Hz line noise
        Y = np.fft.rfft(eeg - np.nanmean(eeg))
        P = np.abs(Y) ** 2
        freqs = np.fft.rfftfreq(len(eeg), d=1 / fs)

        P_50Hz = np.sum(P[(freqs > 48) & (freqs < 52)]) / np.sum(P)

        if P_50Hz > percent_50_hz:
            bad.append(ich)
            noisy_ch.append(ich)
            continue

    # Remove channels with unusually high standard deviation
    median_std = np.nanmedian(all_std)
    higher_std = np.where(all_std > mult_std * median_std)[0]

    for ch in higher_std:
        if ch not in bad:
            bad.append(ch)

    bad = np.array(np.unique(bad), dtype=int)

    logging.info(
        f'kh{ppt.kh_id:03d} | Prep: {ppt.prep} | Heuristic bad channels - removed: {ppt.exp.channels[bad]}'
    )

    # Store removed channels
    ppt.noise_outliers = ppt.exp.channels[bad]

    # Store reasons for removal
    ppt.noise_outlier_details = {
        "nans": ppt.exp.channels[np.asarray(nan_ch, dtype=int)],
        "zeros": ppt.exp.channels[np.asarray(zero_ch, dtype=int)],
        "high_voltage": ppt.exp.channels[np.asarray(high_ch, dtype=int)],
        "high_variance": ppt.exp.channels[np.asarray(high_var_ch, dtype=int)],
        "50Hz_noise": ppt.exp.channels[np.asarray(noisy_ch, dtype=int)],
        "high_std": ppt.exp.channels[higher_std],
    }

    # Remove bad channels
    ppt.exp.eeg = np.delete(ppt.exp.eeg, bad, axis=1)
    ppt.exp.channels = np.delete(ppt.exp.channels, bad)

    return ppt

def local_outlier_factor(ppt, k=None, **kwargs):
    '''
    Flags channels if they have different local 
    neighborhoods in power spectral density space.

    i.e. the power characteristics in a channel 
    is different than the other channels.

    Parameters
    ----------
    ppt: Participant instance
    k: None
       amount of channels to include in neighborhood

    Return
    ----------
    ppt: Participant instance

    '''
    
    eeg = ppt.exp.eeg - ppt.exp.eeg.mean(axis=0)
    psd = np.abs(np.fft.rfft(eeg, axis=0))

    lof = LocalOutlierFactor(n_neighbors=k if k else 20)
    outliers = np.where(lof.fit_predict(psd.T)==-1)[0]

    logging.info(f'kh{ppt.kh_id:03d} | Prep: {ppt.prep} | LOF - removed channels: {ppt.exp.channels[outliers]}')

    ppt.noise_outliers = ppt.exp.channels[outliers]

    ppt.exp.eeg = np.delete(ppt.exp.eeg, outliers, axis=1)
    ppt.exp.channels = np.delete(ppt.exp.channels, outliers)

    return ppt


def noise(ppt, k=5, exclude_line_noise=True):

    switch = 'heuristic'

    if switch == 'heuristic':
        return heuristic_bad_chs(ppt)
    elif switch == 'lof':
        return local_outlier_factor(ppt)


if __name__=='__main__':
    random_clusters = np.hstack((np.random.normal(loc=1, scale=1, size=(100, 5)),
                               np.random.normal(loc=4, scale=1, size=(100, 5)),
                               np.random.normal(loc=3, scale=3, size=(100, 2))))

    local_outlier_factor(random_clusters)
