'''
NOTE: MAKE SURE MAIN IS IN YOUR CURRENT WORKING DIRECTORY,
      OTHERWISE ALL IMPORTS WILL FAIL.
'''

# Builtin
import logging
import shutil
from pathlib import Path
from datetime import datetime as dt

# 3th party
import numpy as np

# Local
from libs.helpers.load_yaml import load_yaml
from libs.loading.load import load_task
from libs.preprocessing import go as preprocess
from libs.feature_selection.features_grasp import go as features_grasp
from libs.feature_selection.features_speech import go as features_speech
from libs.univariate import go as univariate
from libs.multivariate import go as multivariate

LOG_DEBUG = False

TASKS = ['grasp'] #['speech', 'grasp']
PREPS = ['RAW', 'RR_CAR', 'RR_CMR', 'RR_ESR', 'RR_LPR', 'RR_BPR', 'RR_ICA'] 
         #Optional data removal: ['WM_ONLY', 'WM_MAJ', 'WM_PROX', 'NOISE']
BANDS = ['beta'] #['broadband']
METHODS = ['allchs', 'selchs', 'selpca']

str_now = dt.now().strftime('%Y%m%d%H%M%S')
log_filename = f'{str_now}_output.log'
#make sure log folder exists
Path('logs').mkdir(parents=True, exist_ok=True)
logging.basicConfig(format='%(levelname)s: %(message)s',
                    level=logging.INFO if not LOG_DEBUG else logging.DEBUG,
                    handlers=[
                        logging.FileHandler(f'logs/{log_filename}'),
                        logging.StreamHandler()])

OUT_PATH = Path(f'./output/{str_now}/')


def setup_debug(ppt, task, cutoff=50000):
    # Cut to cutoff samples to speed up development
    ppt.exp.eeg = ppt.exp.eeg[:cutoff, :]
    ppt.exp.labels = ppt.exp.labels[:cutoff]
    ppt.exp.ts = ppt.exp.ts[:cutoff]

    if task == 'speech':
        cutoff = int((cutoff/ppt.exp.fs)*ppt.exp.audio_fs)
        ppt.exp.audio = ppt.exp.audio[:cutoff]

    return ppt


def features(ppt, band, task):
    '''
    Calls functions that prepares data set
    as appropriate for that task. Output
    should be directly usable by some
    cross validation learning.

    returns:
        data: 2d np.array of [samples x channels]
    '''
    ppt.band = band
    if task == 'speech': return features_speech(ppt)
    if task == 'grasp':  return features_grasp(ppt)


def go():

    for task in TASKS:

        logging.info(f'Current task: {task}'.upper())
        
        ppts = load_task(task)

        for band in BANDS:
            for prep in PREPS:
                for ppt in ppts:

                    if not ppt.xdf_path or not ppt.xdf_loader:
                        continue
                        
                    ppt.load()  # Saves experimental data to a
                                # dataclass in ppt.exp 

                    # Debug mode
                    if False:
                        ppt = setup_debug(ppt, task, cutoff=50000)

                    # Preprocessing
                    ppt = preprocess(ppt, prep)
                    
                    # Skip subject if 0 channels remain
                    if ppt.exp.eeg.shape[1] < 1: 
                        logging.warning(f'{ppt.id} | Band: {band} | Prep: {ppt.prep} | Channels N={ppt.exp.eeg.shape[1]} < 1 | Skipping all analyses')
                        ppt.save(Path(OUT_PATH))
                        continue

                    # Features
                    ppt = features(ppt, band, task)

                    # Univariate
                    ppt = univariate(ppt)

                    # Skip subject if less than 10 channels remain
                    if ppt.exp.eeg.shape[1] < 10: 
                        logging.warning(f'{ppt.id} | Band: {ppt.band} | Prep: {ppt.prep} | Channels N={ppt.exp.eeg.shape[1]} < 10 | Skipping multivariate analyses')
                        ppt.save(Path(OUT_PATH))
                        continue

                    # Multivariate
                    for method in METHODS:
                        ppt = multivariate(ppt, method)

                    # Save results
                    ppt.save(Path(OUT_PATH))
                    
                    logging.info('\n')

    # Move logs from successful run to appropriate folder
    try:
        shutil.copyfile(f'./logs/{log_filename}', OUT_PATH/'log.log')
    except FileNotFoundError:
        logging.warning(f'Output folder <{OUT_PATH}> not found; which likely means no data could be found to process.')

    return


if __name__=='__main__':
    try:
        go()
    except Exception as e:
        logging.error(e)
        raise

# Command to run without stopping when exiting
# nohup /opt/conda/bin/python /home/coder/Preprocessing-sEEG/main.py
