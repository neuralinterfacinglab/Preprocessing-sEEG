from pathlib import Path
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, shapiro, ttest_rel, wilcoxon

from libs.loading.load_results import load_all
import libs.maps as maps

import pandas as pd
import statsmodels.formula.api as smf

ALPHA = 0.05  # stats
ALPHA_SCATTER_PLOTS = 1  # opacity
XLABELS = ['WMR', 'CAR', 'CMR', 'ESR', 'LPR', 'BPR', 'ICR']


def annote_significance(ax, values_combi, p_values: list):

    y_max_values = [np.max(r) for r in values_combi]
    y_min_values = [np.min(r) for r in values_combi]

    x_ticks = ax.get_xticks()

    is_sig =  lambda p_value: ''    if p_value == -1   else \
                              '***' if p_value < 0.001 else \
                              '**'  if p_value < 0.01  else \
                              '*'   if p_value < 0.05  else \
                              'n.s'

    for x_tick, y_max_value, y_min_value, p_value in zip(x_ticks[1:], y_max_values[1:], y_min_values[1:], p_values):
        if y_max_value <= 0.95:
            ax.annotate(f'{is_sig(p_value)}', xy=(x_tick, y_max_value+0.04), ha='center', va='bottom')
        else:
            ax.annotate(f'{is_sig(p_value)}', xy=(x_tick, y_min_value-0.04), ha='center', va='top')

    return ax


def fdr_correction(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]
    

def get_statistics(results: list, task, band) -> list:

    #Keep only participants with no NaNs across any condition
    valid = ~np.isnan(results).any(axis=0)
    print(f"Removed {np.sum(~valid)} participant(s) with missing data")
    results = results[:, valid]

    baseline = results[0]
    methods  = results[1:]
    n_conditions, n_participants = results.shape

    #Main effect
    main = friedmanchisquare(*results)
    print(f'Main effect of {task}-{band}: χ²({n_conditions - 1}) = {main.statistic:.2f}, p = {main.pvalue:.3f}')

    #Posthoc tests
    normal_s = []
    normal_p = []
    posthoc = []
    posthoc_t = []
    posthoc_s = []
    posthoc_p = []
    pvalues_fdr = []
    for i, condition in enumerate(methods, start=1):
        # Perform Shapiro-Wilk test for normality on the differences
        differences = condition-baseline
        stat, p_value = shapiro(differences)
        # Output the results
        print(f"Statistical tests for condition {i}:")
        print(f"  Shapiro-Wilk statistic: {stat:.2f}, p-value: {p_value:.4f}")
        normal_s.append(stat)
        normal_p.append(p_value)
        if p_value > 0.05:
            print(f"  The differences between baseline and condition {i} are normally distributed (fail to reject H0)")
            # Use paired t-test if normally distributed
            t_stat, t_p_value = ttest_rel(baseline, condition)
            print(f"  Paired t-test: t = {t_stat:.2f}, p-value = {t_p_value:.4f}")
            posthoc.append(('t', t_stat, t_p_value))
            posthoc_t.append('t')
            posthoc_s.append(t_stat)
            posthoc_p.append(t_p_value)
        else:
            print(f"  The differences between baseline and condition {i} are not normally distributed (reject H0)")
            # Use Wilcoxon signed-rank test if not normally distributed
            w_stat, w_p_value = wilcoxon(baseline, condition)
            print(f"  Wilcoxon test: W = {w_stat:.0f}, p-value = {w_p_value:.4f}")
            posthoc.append(('W', w_stat, w_p_value))
            posthoc_t.append('W')
            posthoc_s.append(w_stat)
            posthoc_p.append(w_p_value)       
       

    #FDR correction
    pvalues_fdr = fdr_correction([test[2] for test in posthoc])

    #Print posthoc results with FDR adjustment
    for i, (result, p_fdr) in enumerate(zip(posthoc, pvalues_fdr)):
        print(f"Posthoc comparison {i + 1}: {result[0]} = {result[1]:.2f}, n = {n_participants}, unadjusted p = {result[2]:.3f}, FDR-adjusted p = {p_fdr:.3f}")

    return pvalues_fdr


def extract_results(data):

    tasks =   maps.TASKS
    bands =   maps.BANDS
    ppts =    maps.PPTS
    preps =   maps.BASELINE + maps.REFERENCING
    methods = maps.METHODS
    methods = ['allchs','selchs','selpca']

    n_folds = 10

    n_tasks, n_bands, n_ppts, n_preps, n_methods = len(tasks), len(bands), len(maps.PPTS['grasp']), len(preps), len(methods)

    results = np.full((n_tasks, n_bands, n_methods, n_preps, n_ppts, n_folds), fill_value=np.nan)

    for task in tasks:
        for band in bands:
            for prep, method, ppt in product(preps, methods, ppts[task]):
                results[tasks.index(task),
                        bands.index(band),
                        methods.index(method),                        
                        preps.index(prep),
                        ppts[task].index(ppt), :] = data[task][band][prep][ppt][method]['s_te'].ravel()
        
    return results


def plot_panel(ax, results, task, band):

    means, stds = results.mean(axis=-1), results.std(axis=-1)

    values_combi = means

    x_ticks = np.arange(means.shape[0]) + 1

    colors = [maps.get_subj_color_map()[ppt] for ppt in maps.PPTS[task]]

    #violinplot
    parts = ax.violinplot(values_combi.T, positions=x_ticks, showmeans=True, showextrema=True)

    for pc in parts['bodies']:
        pc.set(facecolor='grey', alpha=0.2)
    parts['cmeans'].set_colors('k')
    parts['cbars'].set_alpha(0.5)

    for partname in ('cmins','cmaxes','cbars'):
        vp = parts[partname]
        vp.set_edgecolor('grey')
        vp.set_linewidth(1)

    # #boxplot
    # parts = ax.boxplot(values_combi.T, positions=x_ticks, patch_artist=True, whis=(0,100), showfliers=False)
            
    # for i, pc in enumerate(parts['boxes']):
    #     pc.set(facecolor='grey', alpha=0.2)
    #     parts['medians'][i].set(color='grey', linewidth=1)

    for i, dots in enumerate(means, start=1):
        x_jitter = i + np.linspace(-dots.size/2, dots.size/2, dots.size) * 0.03
        ax.scatter(x_jitter, dots, s=8, c=colors, alpha=ALPHA_SCATTER_PLOTS)

    ax.axhline(np.mean(values_combi[0,:]), linestyle='dashed', color='k', alpha=0.3)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([])
    ax.set_ylim(0.4, 1)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    
    p_values = get_statistics(values_combi, task, band)
    annote_significance(ax, values_combi, p_values)

    return ax


def make(path: Path):

    data = load_all(path=path)

    decoding_results = extract_results(data)

    for method in maps.METHODS:

        print(f'\n--- RUNNING RESULTS FOR METHOD: {method} ---')

        #Figure
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))

        for task, band in product(maps.TASKS, maps.BANDS):
            it, ib, im = maps.TASKS.index(task), maps.BANDS.index(band), maps.METHODS.index(method)
            axs[ib, it] = plot_panel(axs[ib, it], decoding_results[it, ib, im, :, :, :], task, band)
            if ib == 0:
                axs[ib, it].set_title(f'{task.capitalize()}', fontsize='large')
            else:
                axs[ib, it].set_xticklabels(XLABELS, rotation=45, ha='center')
            if it == 0:
                axs[ib, it].set_ylabel(f'{band.capitalize()}\nMean ROC-AUC', fontsize='large')
        
        fig.savefig(f'./figures/{Path(__file__).stem}-{method}.png')
        fig.savefig(f'./figures/{Path(__file__).stem}-{method}.svg')