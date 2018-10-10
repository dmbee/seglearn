'''
This module is for loading time series data sets
'''
# Author: David Burns
# License: BSD

from os.path import dirname
import numpy as np

__all__ = ['load_watch']


def load_watch():
    '''
    Loads some of the 6-axis inertial sensor data from my smartwatch project. The sensor data was
    recorded as study subjects performed sets of 20 shoulder exercise repetitions while wearing a
    smartwatch. It is a multivariate time series.

    The study can be found here: https://arxiv.org/abs/1802.01489

    Returns
    -------
    data : dict
        data['X'] : list, length 140
            | inertial sensor data, each element with shape [n_samples, 6]
            | sampled at 50 Hz
        data['y'] : array, length 140
            target vector (exercise type)
        data['side'] : array, length 140
            the extremity side, 1 = right, 0 = left
        data['subject'] : array, length 140
            the subject (participant) number
        data['X_labels'] : str list, length 6
            ordered labels for the sensor data variables
        data['y_labels'] :str list, length 7
            ordered labels for the target (exercise type)

    Examples
    --------
    >>> from seglearn.datasets import load_watch
    >>> data = load_watch()
    >>> print(data.keys())
    '''
    module_path = dirname(__file__)
    data = np.load(module_path + "/data/watch_dataset.npy").item()
    return data
