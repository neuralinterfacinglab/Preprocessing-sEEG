# Builtin
import logging
from pathlib import Path
from typing import Callable

# 3th party
from pandas import read_excel, read_csv

# Local
import libs.loading.load_speech as load_speech
import libs.loading.load_grasp as load_grasp
from libs.participant import Participant


PATH_DOCS = Path('./documents')
PATH_DATA = Path('./data')


def get_filenames(path_main: Path, extension: str, 
                  keywords: list=[], exclude: list=['_archive']) -> list[Path]:
    ''' Recursively retrieves all files with 'extension', 
    and subsequently filters by given keywords. 

    keywords: list[str,]
        Selects file when substring exists in filename

    exlude: list[str,]
        Removes file if substring in filename or string equals
        any complete parent foldername.
    '''

    if not path_main.exists():
        raise NameError(f'Cannot access {path_main}')

    keywords = extension if len(keywords)==0 else keywords
    extension = f'*.{extension}' if extension[0] != '.' else f'*{extension}'
    files = [path for path in Path(path_main).rglob(extension) \
             if any(kw in path.name for kw in keywords)]

    if any(exclude):
        files = [path for path in files for excl in exclude \
                   if excl not in path.name
                   and excl not in path.parts]
    return files

def get_loader(task: str) -> Callable:
    if task == 'speech': return load_speech.load
    if task == 'grasp':  return load_grasp.load

    raise Exception(f'Cannot find loading function for task={task}')

def load_locations(ppt: Participant) -> Participant:
    path_locations = PATH_DATA/f'{ppt.id}'/'electrode_locations.csv'

    if not path_locations.exists():
        # Could be raise, as electrode locations are essential
        logging.info(f'{ppt.id} | Cannot find electrode locations')
        return ppt

    l = read_csv(path_locations)
    ppt.electrode_mapping = dict(zip(l['electrode_name_1'], l['location']))
    ppt.PTD = dict(zip(l['electrode_name_1'], l['PTD']))

    return ppt

def load_dataset_info(task: str) -> list[Participant]:
    info = read_excel(PATH_DOCS/'dataset.xlsx').fillna(0)

    ppts = [Participant(row) for _, row in info[info[task.capitalize()]==1].iterrows()]

    return ppts

def load_task(task: str) -> list[Participant]:
    ''' Loads information and prepares for loading
        data for all participants for given task.

        returns
            list[Partipant]: list of instances of
                             participants.
    '''

    # Load info per participant
    ppts = load_dataset_info(task)
    ppts = [load_locations(ppt) for ppt in ppts]

    # Correct for specific filenames
    filename = 'singlewords' if task == 'speech' else task
    
    # Attach datapath + associated loader on ppt instance
    for ppt in ppts:
        ppt_path = PATH_DATA/f'{ppt.id}'
        try:
            ppt.xdf_path = get_filenames(ppt_path, 'xdf', [filename])[0]  # If multiple measurements, select first one
            ppt.xdf_loader = get_loader(task)
            ppt.task = task
        except Exception:
            logging.info(f'{ppt.id} | Failed to setup loading for {filename}')

    return ppts
