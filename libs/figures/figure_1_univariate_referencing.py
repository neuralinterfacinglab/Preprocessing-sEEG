import sys
from pathlib import Path
from itertools import product

# 3th party
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import friedmanchisquare, shapiro, ttest_rel, wilcoxon

# import libs.loading.load_results as load_all  # Run from make_all_figures.py
from libs.loading.load_results import load_all
import libs.maps as maps

XLABELS = ['WMR', 'CAR', 'CMR', 'ESR', 'LPR', 'BPR', 'ICR']
ALPHA_SCATTER_PLOTS = 1


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
    n = len(baseline)

    #Main effect
    main = friedmanchisquare(*results)
    print(f'Main effect of {task}-{band}: χ²({n - 1}) = {main.statistic:.2f}, p = {main.pvalue:.3f}')

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
    pvalues_fdr = fdr_correction([value for value in posthoc_p])

    #Print posthoc results with FDR adjustment
    for i, (result, p_fdr) in enumerate(zip(posthoc, pvalues_fdr)):
        print(f"Posthoc comparison {i + 1}: {result[0]} = {result[1]:.2f}, n = {n}, unadjusted p = {result[2]:.3f}, FDR-adjusted p = {p_fdr:.3f}")

    return pvalues_fdr

def annote_significance_above_max_y_value(ax, y_values: list, p_values: list):

    x_ticks = ax.get_xticks()

    is_sig =  lambda p_value: ''    if p_value == -1   else \
                              '***' if p_value < 0.001 else \
                              '**'  if p_value < 0.01  else \
                              '*'   if p_value < 0.05  else \
                              'n.s'

    for x_tick, y_value, p_value in zip(x_ticks[1:], y_values[1:], p_values):
        ax.annotate(f'{is_sig(p_value)}', xy=(x_tick, y_value+0.05), ha='center', va='bottom')

    return ax

def extract_icc(handle):
    # input: data[task][band][prep][ppt]
    icc_all = handle['icc']
    #calculate mean icc across channels for each fold
    icc = [np.mean(np.abs(icc_all[f,:,:][np.triu_indices(icc_all.shape[1], k=1)])) for f in range(icc_all.shape[0])]
    return icc
 
def extract_corr(handle):
    # input: data[task][band][prep][ppt]
    pbc_all = handle['pbc']
    pbc = np.max(np.abs(pbc_all[:,:,0]), axis=1)
    return pbc

def extract_corr_mean(handle):
    # input: data[task][band][prep][ppt]
    pbc_all = handle['pbc']
    pbc = np.mean(np.abs(pbc_all[:,:,0]), axis=1)
    return pbc

def plot_panel(ax, data, func):
    # func = function to extract data
    tasks =   maps.TASKS
    bands =   maps.BANDS
    ppts =    maps.PPTS
    preps =   maps.BASELINE + maps.REFERENCING

    n_tasks, n_bands, n_ppts, n_preps, n_folds = len(tasks), len(bands), len(maps.PPTS['grasp']), len(preps), 10

    results = np.full((n_tasks, n_bands, n_preps, n_ppts, n_folds), fill_value=np.nan)

    for task in tasks:
        for band in bands:
            for prep, ppt in product(preps, ppts[task]):
                results[tasks.index(task),
                        bands.index(band),
                        preps.index(prep),
                        ppts[task].index(ppt), :] = func(data[task][band][prep][ppt])
                      

    for i_task, task in enumerate(tasks):
        for i_band, band in enumerate(bands):

            colors = [maps.get_subj_color_map()[ppt] for ppt in maps.PPTS[task]]

            x_ticks = np.arange(len(preps)) + 1

            values = results[i_task, i_band, :, :]
            means, stds = values.mean(axis=-1), values.std(axis=-1)
            values_combi = means

            #violinplot
            parts = ax[i_band, i_task].violinplot(values_combi.T, positions=x_ticks, showmeans=True, showextrema=True)

            for pc in parts['bodies']:
                pc.set(facecolor='grey', alpha=0.2)
            parts['cmeans'].set_colors('k')
            parts['cbars'].set_alpha(0.5)

            for partname in ('cmins','cmaxes','cbars'):
                vp = parts[partname]
                vp.set_edgecolor('grey')
                vp.set_linewidth(1)

            # #boxplot
            # parts = ax[i_band, i_task].boxplot(values_combi.T, positions=x_ticks, whis=(0,100), patch_artist=True, showfliers=False)
            
            # for i, pc in enumerate(parts['boxes']):
            #     pc.set(facecolor='grey', alpha=0.2)
            #     parts['medians'][i].set(color='grey', linewidth=1)
            
            for i, dots in enumerate(means, start=1):
                x_jitter = i + np.linspace(-dots.size/2, dots.size/2, dots.size) * 0.03
                ax[i_band, i_task].scatter(x_jitter, dots, s=8, c=colors, alpha=ALPHA_SCATTER_PLOTS)

            ax[i_band, i_task].axhline(np.mean(values_combi[0]), linestyle='dashed', color='k', alpha=0.3)
            ax[i_band, i_task].set_xticks(x_ticks)
            ax[i_band, i_task].set_xticklabels([])
            ax[i_band, i_task].set_ylim(0, 0.8)
            ax[i_band, i_task].spines.right.set_visible(False)
            ax[i_band, i_task].spines.top.set_visible(False)
            if i_band == 0:
                ax[i_band, i_task].set_title(f'{task.capitalize()}', fontsize='large')
            else:
                ax[i_band, i_task].set_xticklabels(XLABELS, rotation=45, ha='center')

            # Statistics
            p_values = get_statistics(values_combi, task, band)
            annote_significance_above_max_y_value(ax[i_band, i_task], [np.max(r) for r in values_combi], p_values)
      
    return ax

def make(path: Path):

    metrics = ['icc', 'pbc', 'pbcmean']

    data = load_all(path=path)

    for metric in metrics:     

        print(f'\n--- RUNNING RESULTS FOR METRIC: {metric} ---')

        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        fig.show()

        if metric=='icc':
            axs_icc = plot_panel(axes, data, extract_icc)
            axs_icc[0,0].set_ylabel('Beta\nMean interchannel correlation', fontsize='large')
            axs_icc[1,0].set_ylabel('Broadband\nMean interchannel correlation', fontsize='large')

        elif metric=='pbc':
            axs_pbc = plot_panel(axes, data, extract_corr)
            axs_pbc[0,0].set_ylabel('Beta\nMaximum task correlation', fontsize='large')
            axs_pbc[1,0].set_ylabel('Broadband\nMaximum task correlation', fontsize='large')

        elif metric=='pbcmean':
            axs_pbc = plot_panel(axes, data, extract_corr_mean)
            axs_pbc[0,0].set_ylabel('Beta\nMean task correlation', fontsize='large')
            axs_pbc[1,0].set_ylabel('Broadband\nMean task correlation', fontsize='large')                

        # plt.tight_layout()
        
        fig.savefig(f'./figures/{Path(__file__).stem}-{metric}.png')
        fig.savefig(f'./figures/{Path(__file__).stem}-{metric}.svg')
