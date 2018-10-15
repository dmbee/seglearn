'''
This module has two classes for splitting time series data temporally - where train/test or fold
splits are created within each of the time series' in the time series data. This splitting approach
is for evaluating how well the algorithm performs on segments drawn from the same time series but
excluded from the training set. The performance from this splitting approach should be similar to
performance on the training data so long as the data in each series is relatively uniform.

Note that splitting along the temporal axis violates the assumption of independence between train
and test samples, as successive samples in a sequence or series are not iid. However, temporal
splitting is still useful in certain cases such as for the analysis of a single sequence / series.
'''
# Author: David Burns
# License: BSD

import numpy as np

from .util import check_ts_data, get_ts_data_parts
from .base import TS_Data


class TemporalKFold(object):
    '''
    K-fold iterator variant for temporal splitting of time series data

    The time series' are divided in time with no overlap, and are balanced.

    By splitting the time series', the number of samples in the data set is changed and so new
    arrays for the data and target are returned by the ``split`` function in addition to the
    iterator.

    Parameters
    ----------
    n_splits : int > 1
        number of folds

    Examples
    --------
    >>> from seglearn.split import TemporalKFold
    >>> from seglearn.datasets import load_watch
    >>> data = load_watch()
    >>> splitter = TemporalKFold(n_splits=4)
    >>> X, y, cv = splitter.split(data['X'], data['y'])

    '''

    def __init__(self, n_splits=3):
        if n_splits < 2:
            raise ValueError("TemporalKFold: n_splits must be >= 2 (set to %d)" % n_splits)
        self.n_splits = n_splits

    def split(self, X, y):
        '''
        Splits time series data and target arrays, and generates splitting indices

        Parameters
        ----------
        X : array-like, shape [n_series, ...]
           Time series data and (optionally) contextual data
        y : array-like shape [n_series, ]
            target vector

        Returns
        -------
        X : array-like, shape [n_series * n_splits, ]
            Split time series data and contextual data
        y : array-like, shape [n_series * n_splits]
            Split target data
        cv : list, shape [2, n_splits]
            Splitting indices
        '''

        check_ts_data(X, y)
        Xt, Xc = get_ts_data_parts(X)
        Ns = len(Xt)
        Xt_new, y_new = self._ts_slice(Xt, y)

        if Xc is not None:
            Xc_new = np.concatenate([Xc] * self.n_splits)
            X_new = TS_Data(Xt_new, Xc_new)
        else:
            X_new = np.array(Xt_new)

        cv = self._make_indices(Ns)

        return X_new, y_new, cv

    def _ts_slice(self, Xt, y):
        ''' takes time series data, and splits each series into temporal folds '''
        Ns = len(Xt)
        Xt_new = []
        for i in range(self.n_splits):
            for j in range(Ns):
                Njs = int(len(Xt[j]) / self.n_splits)
                Xt_new.append(Xt[j][(Njs * i):(Njs * (i + 1))])
        Xt_new = np.array(Xt_new)

        if len(np.atleast_1d(y[0])) == len(Xt[0]):
            # y is a time series
            y_new = []
            for i in range(self.n_splits):
                for j in range(Ns):
                    Njs = int(len(y[j]) / self.n_splits)
                    y_new.append(y[j][(Njs * i):(Njs * (i + 1))])
            y_new = np.array(y_new)
        else:
            # y is contextual to each series
            y_new = np.concatenate([y for i in range(self.n_splits)])

        return Xt_new, y_new

    def _make_indices(self, Ns):
        ''' makes indices for cross validation '''
        N_new = int(Ns * self.n_splits)

        test = [np.full(N_new, False) for i in range(self.n_splits)]
        for i in range(self.n_splits):
            test[i][np.arange(Ns * i, Ns * (i + 1))] = True
        train = [np.logical_not(test[i]) for i in range(self.n_splits)]

        test = [np.arange(N_new)[test[i]] for i in range(self.n_splits)]
        train = [np.arange(N_new)[train[i]] for i in range(self.n_splits)]

        cv = list(zip(train, test))
        return cv


def temporal_split(X, y, test_size=0.25):
    '''
    Split time series or sequence data along the time axis.
    Test data is drawn from the end of each series / sequence

    Parameters
    ----------
    X : array-like, shape [n_series, ...]
       Time series data and (optionally) contextual data
    y : array-like shape [n_series, ]
        target vector
    test_size : float
        between 0 and 1, amount to allocate to test

    Returns
    -------
    X_train : array-like, shape [n_series, ]
    X_test : array-like, shape [n_series, ]
    y_train :  array-like, shape [n_series, ]
    y_test :  array-like, shape [n_series, ]

    '''

    if test_size <= 0. or test_size >= 1.:
        raise ValueError("temporal_split: test_size must be >= 0.0 and <= 1.0"
                         " (was %.1f)" %test_size)

    Ns = len(y)  # number of series
    check_ts_data(X, y)
    Xt, Xc = get_ts_data_parts(X)

    train_size = 1. - test_size

    train_ind = [np.arange(0, int(train_size * len(Xt[i]))) for i in range(Ns)]
    test_ind = [np.arange(len(train_ind[i]), len(Xt[i])) for i in range(Ns)]

    X_train = [Xt[i][train_ind[i]] for i in range(Ns)]
    X_test = [Xt[i][test_ind[i]] for i in range(Ns)]

    if Xc is not None:
        X_train = TS_Data(X_train, Xc)
        X_test = TS_Data(X_test, Xc)

    if len(np.atleast_1d(y[0])) == len(Xt[0]):
        # y is a time series
        y_train = [y[i][train_ind[i]] for i in range(Ns)]
        y_test = [y[i][test_ind[i]] for i in range(Ns)]
    else:
        # y is contextual
        y_train = y
        y_test = y

    return X_train, X_test, y_train, y_test
