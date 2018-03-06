import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class Segment(BaseEstimator, TransformerMixin):
    ''' separate predict and transform functions with different overlap '''
    def __init__(self, width = 100, overlap = 0.5):
        self.width = width
        self.overlap = overlap
        self.f_labels = None

    def fit(self, X, y = None):
        self._reset()

        assert self.width > 0
        assert self.overlap >= 0 and self.overlap <= 1

        self.step = int(self.width * (1. - self.overlap))
        self.step = self.step if self.step >= 1 else 1
        return self

    def _reset(self):
        if hasattr(self, 'step'):
            del self.step

    def transform(self, X):
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
    segments time_series with sliding windows
    :param time_series: numpy array shape (T,)
    :param step: number of data points to advance for each window
    :param width: length of window in samples
    :return: segmented time_series, numpy array shape (N, width)
    '''
    w = np.hstack(time_series[i:1 + i - width or None:step] for i in range(0, width))
    return w.reshape((int(len(w)/width),width),order='F')

def sliding_tensor(mv_time_series, width, step):
    '''
    segments multivariate time series with sliding windows
    :param mv_time_series: numpy array shape (T, D) with D time series variables
    :param width: length of window in samples
    :param step: number of data points to advance for each window
    :return: multivariate temporal tensor, numpy array shape (N, W, D)
    '''
    D = mv_time_series.shape[1]
    data = [sliding_window(mv_time_series[:, j], width, step) for j in range(D)]
    return np.stack(data, axis = 2)