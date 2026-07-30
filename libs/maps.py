from matplotlib import cm

TASKS = ['grasp', 'speech']

BANDS = ['beta', 'broadband']

PPTS = {
    'speech': ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10'],
    'grasp':  ['sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10', 'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15']
}

PREPS = ['RAW', 'RR_CAR', 'RR_CMR', 'RR_ESR', 'RR_LPR', 'RR_BPR', 'RR_ICR', 'NOISE', 'WM_ONLY', 'WM_MAJ', 'WM_PROX']

BASELINE = ['RAW']
REFERENCING = ['RR_CAR', 'RR_CMR', 'RR_ESR', 'RR_LPR', 'RR_BPR', 'RR_ICR']
EXCLUSION = ['NOISE', 'WM_ONLY', 'WM_MAJ', 'WM_PROX']

METHODS = ['allchs', 'selchs', 'selpca']

def get_id_map():
    return {'1': 'sub-01',  # S
            '2': 'sub-02',  # S
            '3': 'sub-03',  # S
            '4': 'sub-04',  # S
            '5': 'sub-05',  # S
            '6': 'sub-06',  # S & G
            '7': 'sub-07',  # S & G
            '8': 'sub-08',  # S & G
            '9': 'sub-09',  # S & G
            '10': 'sub-10',  # S & G
            '11': 'sub-11',  # G
            '12': 'sub-12',  # G
            '13': 'sub-13',  # G
            '14': 'sub-14',  # G
            '15': 'sub-15'}  # G

def get_subj_color_map():

    cmap = cm.get_cmap('tab20')

    ids = get_id_map().values() #color mapping based on sub code order!
    
    return {id_: cmap(i) for i, id_ in enumerate(ids)}

def get_method_color_map():

    cmap = cm.get_cmap('Set3')

    ids = sorted(get_method_map.values()) #use sub code
    
    return {id_: cmap(i) for i, id_ in enumerate(ids)}

def get_method_map():
    return {'RAW':     'Raw',
            'RR_CAR':  'Common average',
            'RR_CMR':  'Common median',
            'RR_ESR':  'Electrode shaft',
            'RR_LPR':  'Laplacian',
            'RR_BPR':  'Bipolar',
            'WM_MAJ':  'Majority white matter',
            'WM_ONLY': 'Only white matter',
            'WM_PROX': 'Any white matter',
            'NOISE':   'Noise removal'}

def get_method_labels():
    return {'RAW':     'Raw',
            'RR_CAR':  'CAR',
            'RR_CMR':  'CMR',
            'RR_ESR':  'ESR',
            'RR_LPR':  'LPR',
            'RR_BPR':  'BPR',
            'WM_MAJ':  'WM_maj',
            'WM_ONLY': 'WM_only',
            'WM_PROX': 'WM_any',
            'NOISE':   'Noise'}

if __name__=='__main__':
    print(get_id_map())
    print(get_subj_color_map())