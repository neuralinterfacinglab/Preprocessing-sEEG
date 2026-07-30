# sEEG preprocessing

## Install
Python version 3.10.x <

`conda create --name <your_env_name> python=3.10.4` \
`conda activate <your_env_name>` \
`pip install -r requirements.txt` \

Make sure to set this repo as your current working directory, otherwise local imports will fail.\
`python main.py`


### Filestructure

- ./data: Place here raw data to be used for analysis. Data associated with the paper can be downloaded from [10.17605/OSF.IO/867AG](https://doi.org/10.17605/OSF.IO/867AG).
- ./documents: Related files (metadata).
- ./figures: Figures outputted from the scripts.
- ./libs: Python files that are useful for the main scripts. For example: data loading, spectral band extraction, re-referencing methods, evaluation metrics.
- ./logs: Log files for each run of the main script.
- ./output/*: Folder to direct any outputs from the code. The code creates a subfolder for each run of a script that generates a (collection of) files.
