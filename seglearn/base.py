'''
This module has some base classes for time series data
'''

# Author: David Burns
# License: BSD

import numpy as np

__all__ = ['TS_Data']


class TS_Data(object):
    '''
    Iterable/indexable class for time series data with context data
    Numpy arrays are sufficient time series data alone is needed

    Parameters
    ----------
    ts_data : array-like, shape (N, )
        time series data
    context_data : array-like (N, )
        contextual data

    '''

    def __init__(self, ts_data, context_data):
        N = len(ts_data)
        self.ts_data = np.atleast_1d(ts_data)
        self.context_data = np.atleast_1d(context_data)
        self.index = 0
        self.N = N
        self.shape = [N]  # need for safe_indexing with sklearn

    def __iter__(self):
        return self

    def __getitem__(self, indices):
        return TS_Data(self.ts_data[indices], self.context_data[indices])

    def __next__(self):
        if self.index == self.N:
            raise StopIteration
        self.index = self.index + 1
        return TS_Data(self.ts_data[self.index], self.context_data[self.index])

    def __len__(self):
        return self.N
