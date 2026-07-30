import pickle
from pathlib import Path

def is_in(to_check, part):
    if to_check:
       return part in to_check
    else:
        return True

def get_result_files(path:  Path,
                     tasks: list=[],
                     bands: list=[],
                     preps: list=[],
                     ppts:  list=[]):
    '''
    Helper functions that retrieves all files given
    specific requests. Useful for selecting the right 
    data for figures.
    
    Leave the list empty to retrieve all files 
    '''

    all_paths = path.glob('**/*')
    all_files = [p for p in all_paths if p.is_file() and p.suffix != '.log']
 
    files = []
    for file in all_files:
        parts = file.parts
        if is_in(tasks, parts[-4]) and \
           is_in(bands, parts[-3]) and \
           is_in(preps, parts[-2]) and \
           is_in(ppts, file.stem):
           files += [file]

    return files

TASK, BAND, PREP = -4, -3, -2
def load_all(path=r'./output', tasks=[], bands=[], preps=[], ppts=[]):
    paths = get_result_files(Path(path), tasks=tasks, bands=bands,
                                    preps=preps, ppts=ppts)

    data = {}
    for path in paths:
        
        if (task := path.parts[TASK]) not in data:
            data[task] = {}
        
        if (band := path.parts[BAND]) not in data[task]:
            data[task][band] = {}

        if (prep := path.parts[PREP]) not in data[task][band]:
            data[task][band][prep] = {}

        data[task][band][prep].update({path.stem: load_pkl(path)})

    return data

def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data

if __name__ == '__main__':


    path = Path('./output/20241206160202/')
    tasks = ['speech']
    bands = ['broadband']
    preps = ['RAW'] #RR_CAR', 'RR_LPR']
    ppts  = []
    paths = get_result_files(path,
                             tasks=tasks,
                             bands=bands,
                             preps=preps,
                             ppts=ppts)
    for path in paths:
        data = load_pkl(path)

    print('Done')
