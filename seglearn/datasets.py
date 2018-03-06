import numpy as np
from os.path import dirname


def load_watch():
    module_path = dirname(__file__)
    data = np.load(module_path + "/data/watch_dataset.npy").item()
    return data


