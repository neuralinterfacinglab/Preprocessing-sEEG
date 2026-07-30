from dataclasses import dataclass

import numpy as np

'''
Convenient way to store related information,
can contain methods.

'''

@dataclass
class Experiment:
    task: str
    eeg: np.array
    labels: np.array
    channels: np.array
    fs: float
    ts: np.array
    audio: np.array
    audio_fs: int

