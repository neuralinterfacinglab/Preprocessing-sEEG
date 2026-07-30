import logging
import pickle

from libs.loading.experiment import Experiment

class Participant:

    def __init__(self, info) -> None:
        self.id = info[0]
        self.age = info[1]
        self.sex = info[2]
        self.ref = info[3]
        self.gnd = info[4]        
        
        self.xdf_path = None
        self.xdf_loader = None
        self.exp = None
        self.path_output = None

        self.electrode_mapping = {}
        self.PTD = {}
        self.noise_outliers = []        

        self.task = None
        self.band = None
        self.prep = None

        #univariate
        self.icc = None
        self.pbc = None
        
        #multivariate
        self.allchs = None
        self.selchs = None
        self.selpca = None

    def __repr__(self):
        return f'{self.__class__.__name__}(id={self.id}, task={self.task})'

    def load(self):
        if not self.xdf_path or not self.xdf_loader:
            raise Exception(f'{self.id} | Failed loading data -> No path or loader specificied.')

        # For now not saved in instance to prevent memory issues when loading
        # several participants subsequently
        data = self.xdf_loader(self.xdf_path)
        data = data + (None, None) if len(data) == 5 else data  # Might need to generalise
        self.exp = Experiment(self.task, *data)

    def unload(self):
        pass

    def get_results_dict(self, model):

        r = {k:v for k, v in model.__dict__.items() if k != 'clf'}
        
        if 'pca' in model.clf.named_steps:
            pca = model.clf.named_steps['pca']
            r['pca'] = {'components': pca.components_,
                        'expl_var':   pca.explained_variance_}
        return r

    def save(self, path):

        self.path_output = path/f'{self.task}/{self.band}/{self.prep}/'
        self.path_output.mkdir(parents=True, exist_ok=True)

        clf_params = lambda x: [{'name': step[1].__str__().split('(')[0]} | step[1].get_params() \
                                    for step in x.clf.steps]
        
        results = {'settings': {'ppt': {'id':     self.id,
                                        'age':    self.age,
                                        'sex':    self.sex,
                                        'ref':    self.ref,
                                        'gnd':    self.gnd},
                                'exp': {'task':   self.task,
                                        'band':   self.band,
                                        'prep':   self.prep,
                                        'fs':     self.exp.fs,
                                        'chlist': self.exp.channels,
                                        'chexcl': self.exp.excluded,
                                        'chmap':  self.electrode_mapping,
                                        'PTD':    self.PTD,
                                        'chnoise': self.noise_outliers},
                                'learners': {'allchs':  clf_params(self.allchs)  if self.allchs else None,
                                             'selchs': clf_params(self.selchs) if self.selchs else None,
                                             'selchs': clf_params(self.selpca) if self.selpca else None}},
                    'icc':      self.icc,
                    'pbc':      self.pbc,  
                    'allchs':   self.get_results_dict(self.allchs) if self.allchs else None,
                    'selchs':   self.get_results_dict(self.selchs) if self.selchs else None,
                    'selpca':   self.get_results_dict(self.selpca) if self.selpca else None}

        with open(f'{self.path_output}/{self.id}.pkl', 'wb') as f:
            pickle.dump(results, f)

        logging.info(f'{self.id} | Saved to {f.name}')

if __name__=='__main__':
    p = Participant((1, 40, 'm', 'R10', 'L5'))
    print(p)