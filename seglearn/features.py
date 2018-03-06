'''
This module is for calculating hand-coded (eg statistical, heuristic, etc) from a segmented time series.
'''
# Author: David Burns
# License: BSD

import numpy as np
import scipy.stats as stats
from multiprocessing import Pool
from multiprocessing import cpu_count
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class SegFeatures(BaseEstimator, TransformerMixin):
    '''
    A transformer for calculating hand-coded features from a segmented time series

    Parameters
    ----------
    features : dict, optional
        Dictionary of functions for calculating features from a segmented time series.
        Each function in the dictionary is specified to compute features for all variables
        in the segmented time series along axis 1 (the segment) eg:
            >>> def mean_fun(X):
            >>>    F = np.mean(X, axis = 1)
            >>>    return(F)
            X : array-like shape [n_samples, segment_width, n_variables]
            F : array-like [n_samples, n_variables]

        If features is not specified, a default feature dictionary will be used (see _base_features)

    corr_features : bool, optional
        If true, correlation features will be computed between the variables of a multi-variate time series

    multithread: bool, optional
        If true, will use multithreading to compute the features
    '''
    def __init__(self, features = None, corr_features = False, multithread = False):
        self.multithread = multithread
        self.corr_features = corr_features
        self.features = features
        if features is None:
            self.features = self._base_features()

    def fit(self, X, y = None, x_labels = None):
        '''
        Fit the transform Parameters
        ----------
        X : segmented time-series data, array-like shape [n_series, ...]
            Can be a list, numpy array or numpy recarray of segmented time series'.

            Each element of X is array-like shape [n_samples, segment_width, n_variables]
            n_samples can be different for each element of X
            segment_width and n_variables must be constant

            If X is a recarray, the time series data must have name "ts".
            Static variables associated with each series can be named arbitrarily.
        y : None
            There is no need of a target in a transformer, yet the pipeline API requires this parameter.
        x_labels : array-like, optional
            List or array of strings which are labels for the variables in the time series data

        Returns
        -------
        self : object
            Returns self.
        '''

        self._reset()
        if type(X) is np.recarray:
            s_labels = [s for s in X.dtype.names if s != 'ts']
        else:
            s_labels = []
        self.f_labels = self._generate_feature_labels(x_labels, s_labels)
        return self

    def transform(self, X):
        '''
        Transform the segmented time series data into feature data.
        If X is a recarray with time series and static data, the static data is included
        with the returned feature data.

        Parameters
        X : segmented time-series data, array-like shape [n_series, ...]
            Can be a list, numpy array or numpy recarray of segmented time series'.

            Each element of X is array-like shape [n_samples, segment_width, n_variables]
            n_samples can be different for each element of X
            segment_width and n_variables must be constant

            If X is a recarray, the time series data must have name "ts".
            Static variables associated with each series can be named arbitrarily.

        Returns
        -------
        X_new : array shape [n_series, ...]
            The returned feature data, including features computed from the time-series' and any static variables

            Each element in X_new is array-like shape [n_samples, n_features]
        '''
        check_is_fitted(self,'f_labels')
        N = len(X)
        if type(X) is np.recarray:
            arglist = self._recarray_arglist(X)
        else:
            arglist = self._arglist(X)

        if self.multithread == True:
            pool = Pool(cpu_count())
            X_new = pool.map(_feature_thread, arglist)
        else:
            X_new = []
            for i in range(N):
                X_new.append(_feature_thread(arglist[i]))
        return X_new

    def _reset(self):
        ''' Resets internal data-dependent state of the transformer. __init__ parameters not touched. '''
        if hasattr(self, 'f_labels'):
            del self.f_labels

    def _base_features(self):
        ''' Returns data dictionary of some basic features that can be calculated for segmented time series data '''
        uvf = {'mean': mean,
               'std': std,
               'var': var,
               'mnx': mean_crossings}
        return uvf

    def _generate_feature_labels(self, x_labels, s_labels):
        '''
        Generates string feature labels
        Parameters
        ----------
        x_labels : array like shape [n_time_series_variables, ]
            string labels for the variables in the time series
        s_labels : array like shape [n_static_features,]
            string labels for the static features

        Returns
        -------
        f_labels : array like shape [n_features,]
            string feature labels
        '''
        if x_labels is None:
            return None

        f_labels = []
        D = len(x_labels)
        fts_keys = list(self.features.keys())

        for i in range(len(self.features)):
            for j in range(D):
                f_labels += [fts_keys[i] + "_" + x_labels[j]]

        if self.corr_features is True:
            f_labels += self._corr_labels(x_labels)
        f_labels += s_labels

        return f_labels

    def _corr_labels(self, x_labels):
        ''' creates labels for the pearson correlations '''
        D = len(x_labels)
        t = np.triu_indices(D, k=1)  # dont include diag
        labels = []
        for i in range(len(t[0])):
            labels.append("r_" + x_labels[t[0][i]] + x_labels[t[1][i]])
        return labels

    def _arglist(self, X):
        ''' generates argument list for threading feature computation from array-like time-series data '''
        N = len(X)
        return [[{'ts':X[i]}, list(self.features.values()), self.corr_features] for i in range(N)]


    def _recarray_arglist(self, X):
        ''' generates argument list for threading feature computation from rec-array time-series data '''
        N = len(X)
        return [[X[i], list(self.features.values()), self.corr_features] for i in range(N)]

def _feature_thread(args):
    ''' helper function for threading '''
    return _compute_features(*args)

def _compute_features(X, features, corr_feats):
    '''
    Computes features for a segmented time series
    Parameters
    ----------
    X : recarray element or dict
        segmented time series data keyed ['ts']
        and any other static data associated for the series
    features : dict
        dictionary of feature methods
    corr_feats : bool
        specifies if correlational features should be computed

    Returns
        fts : array-like shape [n_samples, n_features]
        computed features
    -------

    '''
    N = X['ts'].shape[0]
    #W = X.shape[1]
    D = X['ts'].shape[2]

    Nuv = len(features)
    fts = np.zeros((N, Nuv * D))

    # todo: change this to use append, and concatenate along columns
    # we'll need number of variables for each for the labels, maybe by running it once and checking width this is a good idea at the fit stage, run on one series,
    # check each n is equal and how many outputs, then we can send names and features sizes.
    # make corr features just another thing in the dictionary
    for i in range(Nuv):
        fts[:, i * D:((i + 1) * D)] = features[i](X['ts'])

    if corr_feats is True:
        corrs = _corr_features(X['ts'])
        fts = np.column_stack([fts, corrs])

    if type(X) is np.recarray:
        hnames = [h for h in X.dtype.names if h != 'ts']
        for h in hnames:
            fts = np.column_stack([fts,np.full(N,X[h])])

    return fts

def _corr_features(X):
    '''
    calculates pearson correlation for all variables in a segmented time series

    Parameters
    ----------
    X : array-like shape [n_samples, segment_width, n_variables]
        segmented time series instance

    Returns
    -------
    r : array-like shape [n_samples, n_correlations]
        correlation feature data
    '''
    D = X.shape[2]
    N = X.shape[0] # not vetorizing this for now, expensive....
    trii = np.triu_indices(D, k=1)
    DD = len(trii[0])
    r = np.zeros((N, DD))
    for i in np.arange(N):
        rmat = np.corrcoef(X[i]) # get the ith window from each signal, result will be DxD
        r[i] = rmat[trii]
    return r

def mean_crossings(X):
    '''
    Computes number of mean crossings for each variable in a segmented time series

    Parameters
    ----------
    X : array-like shape [n_samples, segment_width, n_variables]
        segmented time series instance

    Returns
    -------
    mnx : array-like shape [n_samples, n_variables]
        mean crossings
    '''
    N = X.shape[0]
    D = X.shape[2]
    mnx = np.zeros((N,D))
    for i in range(D):
        pos = X[:,:,i] > 0
        npos = ~pos
        c = (pos[:,:-1] & npos[:,1:]) | (npos[:,:-1] & pos[:,1:])
        mnx[:,i] = np.count_nonzero(c,axis=1)
    return mnx

def mean(X):
    ''' statistical mean for each variable in a segmented time series instance '''
    return np.mean(X, axis = 1)

def std(X):
    ''' statistical standard deviation for each variable in a segmented time series instance '''
    return np.std(X, axis = 1)

def var(X):
    ''' statistical variance for each variable in a segmented time series instance '''
    return np.std(X, axis = 1)

def min(X):
    ''' minimum value for each variable in a segmented time series instance '''
    return np.min(X, axis = 1)

def max(X):
    ''' maximum value for each variable in a segmented time series instance '''
    return np.max(X, axis = 1)

def skew(X):
    ''' skewness for each variable in a segmented time series instance '''
    return stats.skew(X, axis = 1)

def kurt(X):
    ''' kurtosis for each variable in a segmented time series instance '''
    return stats.kurtosis(X, axis = 1)

