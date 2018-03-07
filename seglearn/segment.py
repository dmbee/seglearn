'''
This module is a transformer for performing time series or sequence segmentation with a sliding window
'''
# Author: David Burns
# License: BSD

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class Segment(BaseEstimator, TransformerMixin):
    '''
    Transformer for sliding window segmentation of a time series or sequence

    Parameters
    ----------
    width : int > 0
        width of segments (number of samples)
    overlap : float range [0,1]
        amount of overlap between segments

    Todo
    ----
    separate fit and predict overlap parameters
    '''
    def __init__(self, width = 100, overlap = 0.5):
        self.width = width
        self.overlap = overlap
        self.f_labels = None

    def fit(self, X, y = None):
        '''
        Fit the transform Parameters
        ----------
        X : numpy recarray shape [n_series, ...]
            Segmented time series data must have name 'ts'
            Each element of X['ts'] is array-like shape [n_samples, segment_width, n_variables]
            n_samples can be different for each element of X
            segment_width and n_variables must be constant

            Static variables associated with each series can be named arbitrarily.
        y : None
            There is no need of a target in a transformer, yet the pipeline API requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        '''
        self._reset()
        assert self.width > 0
        assert self.overlap >= 0 and self.overlap <= 1
        self.step = int(self.width * (1. - self.overlap))
        self.step = self.step if self.step >= 1 else 1
        return self

    def _reset(self):
        ''' Resets internal data-dependent state of the transformer. __init__ parameters not touched. '''
        if hasattr(self, 'step'):
            del self.step

    def transform(self, X):
        '''
        Transforms the time series data into segmented time series data

        Parameters
        X : numpy recarray shape [n_series, ...]
            |Time series data is in X['ts']
            |Each element of X['ts'] is array-like shape [n_samples, n_variables]
            |n_samples can be different for each element of X
            |n_variables must be constant for each element of X

        Returns
        -------
        X_new : recarray shape [n_series, ...]
            |The returned segmented time series data is in X_new['ts']
            |Each element of X_new['ts'] is array-like shape [n_samples, width, n_variables]
            |Static variables in X are not changed and returned in X_new

        '''
        check_is_fitted(self, 'step')
        N = len(X)
        if X['ts'][0].ndim > 1:
            Xt = np.array([sliding_tensor(X['ts'][i], self.width, self.step) for i in np.arange(N)])
        else:
            Xt = np.array([sliding_window(X['ts'][i], self.width, self.step) for i in np.arange(N)])
        h_names = [h for h in X.dtype.names if h != 'ts']
        Xh = [X[h] for h in h_names]
        X_new = np.core.records.fromarrays([Xt] + Xh, names=['ts'] + h_names)
        return X_new

def sliding_window(time_series, width, step):
    '''
    Segments univariate time series with sliding window

    Parameters
    ----------
    time_series : array like shape [n_samples]
        time series or sequence
    width : int > 0
        segment width in samples
    step : int > 0
        stepsize for sliding in samples

    Returns
    -------
    w : array like shape [n_segments, width]
        resampled time series segments
    '''
    w = np.hstack(time_series[i:1 + i - width or None:step] for i in range(0, width))
    return w.reshape((int(len(w)/width),width),order='F')

def sliding_tensor(mv_time_series, width, step):
    '''
    segments multivariate time series with sliding window

    Parameters
    ----------
    mv_time_series : array like shape [n_samples, n_variables]
        multivariate time series or sequence
    width : int > 0
        segment width in samples
    step : int > 0
        stepsize for sliding in samples

    Returns
    -------
    data : array like shape [n_segments, width, n_variables]
        segmented multivariate time series data
    '''
    D = mv_time_series.shape[1]
    data = [sliding_window(mv_time_series[:, j], width, step) for j in range(D)]
    return np.stack(data, axis = 2)