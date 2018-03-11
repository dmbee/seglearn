'''
This module has two classes for splitting time series data temporally - where train/test or fold splits are created within each of the time series' in the time series data. This splitting approach is for evaluating how well the algorithm performs on segments drawn from the same time series but excluded from the training set. The performance from this splitting approach should be similar to performance on the training data so long as the data in each series is relatively uniform.
'''
# Author: David Burns
# License: BSD

from .util import check_ts_data, get_ts_data_parts, make_ts_data
import numpy as np

class TemporalKFold():
    '''
    K-fold iterator variant for temporal splitting of time series data

    The time series' are divided in time with no overlap, and are balanced.

    By splitting the time series', the number of samples in the data set is changed and so new arrays for the data and target are returned by the ``split`` function in addition to the iterator.

    Parameters
    ----------
    n_splits : int > 1
        number of folds
    shuffle : bool, default = False
        | if False, the first fold has data from the beginning of each series, the last fold from the end and so on
        | if True, the mapping from part of series to fold is randomized
    random_state : int, default = None
        Randomized may splitting returns different results for each call to ``split``. If you have set ``shuffle`` to True and want the same result with each call to ``split``, set ``random_state`` to an integer.

    Examples
    --------
    >>> from seglearn.split import TemporalKFold
    >>> from seglearn.datasets import load_watch
    >>> data = load_watch()
    >>> splitter = TemporalKFold(n_splits=4)
    >>> X, y, cv = splitter.split(data['X'], data['y'])

    '''

    def __init__(self, n_splits = 3, shuffle = False, random_state=None):
        assert n_splits > 1

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = None # not yet implemented


    def split(self, X, y):
        '''
        Splits time series data and target arrays, and generates splitting indices

        Parameters
        ----------
        X : array-like, shape [n_series, ...]
           Time series data and (optionally) contextual data created as per ``make_ts_data``
        y : array-like shape [n_series]
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

        check_ts_data(X)
        Xt, Xc = get_ts_data_parts(X)
        N = len(Xt)
        Xt_new = self._ts_slice(Xt)

        if Xc is not None:
            Xc_new = np.concatenate([Xc for i in range(self.n_splits)])
        else:
            Xc_new = None

        y_new = np.concatenate([y for i in range(self.n_splits)])

        cv = self._make_indices(N)

        return make_ts_data(Xt_new, Xc_new), y_new, cv

    def _ts_slice(self, Xt):
        ''' takes a time series, splits each one into folds '''
        N = len(Xt)
        X_new = []
        for i in range(self.n_splits):
            for j in range(N):
                Njs = int(len(Xt[j]) / self.n_splits)
                X_new.append(Xt[j][(Njs * i):(Njs * (i + 1))])

        return X_new

    def _make_indices(self, N):
        ''' makes indices for cross validation '''
        N_new = int(N * self.n_splits)

        test = [np.full(N_new, False) for i in range(self.n_splits)]
        for i in range(self.n_splits):
            test[i][np.arange(N * i, N * (i + 1))] = True
        train = [np.logical_not(test[i]) for i in range(self.n_splits)]

        test = [np.arange(N_new)[test[i]] for i in range(self.n_splits)]
        train = [np.arange(N_new)[train[i]] for i in range(self.n_splits)]

        cv = list(zip(train, test))
        return cv


# todo
# inspiration: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# def temporal_train_test_split():
#     pass
#






