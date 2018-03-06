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
        Each function in the dictionary is specified to compute features from a
        multivariate segmented time series along axis 1 (the segment) eg:
            >>> def mean(X):
            >>>    F = np.mean(X, axis = 1)
            >>>    return(F)
            X : array-like shape [n_samples, segment_width, n_variables]
            F : array-like [n_samples, n_features]
            The number of features returned (n_features) must be >= 1

        If features is not specified, a default feature dictionary will be used (see base_features)

    multithread: bool, optional
        If true, will use multithreading to compute the features
    '''
    def __init__(self, features = None, multithread = False):
        self.multithread = multithread
        self.features = features
        if features is None:
            self.features = self.base_features()

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
        self.f_labels = self._generate_feature_labels(X)
        return self

    def transform(self, X):
        '''
        Transform the segmented time series data into feature data.
        If static data is included in X, it is returned with the feature data.

        Parameters
        X : numpy recarray shape [n_series, ...]
            Segmented time series data must have name 'ts'
            Each element of X['ts'] is array-like shape [n_samples, segment_width, n_variables]
            n_samples can be different for each element of X
            segment_width and n_variables must be constant

        Returns
        -------
        X_new : array shape [n_series, ...]
            The returned feature data, including features computed from the time-series' and any static variables
            Each element in X_new is array-like shape [n_samples, n_features]
        '''
        check_is_fitted(self,'f_labels')
        N = len(X)
        arglist = [[X[i], list(self.features.values())] for i in range(N)]

        if self.multithread == True:
            pool = Pool(cpu_count())
            X_new = pool.map(_feature_thread, arglist)
        else:
            X_new = []
            for i in range(N):
                X_new.append(_feature_thread(arglist[i]))

        return np.core.records.fromarrays([X_new], names='ts')

    def _reset(self):
        ''' Resets internal data-dependent state of the transformer. __init__ parameters not touched. '''
        if hasattr(self, 'f_labels'):
            del self.f_labels

    def base_features(self):
        ''' Returns data dictionary of some basic features that can be calculated for segmented time series data '''
        features = {'mean': mean,
               'std': std,
               'var': var,
               'mnx': mean_crossings}
        return features

    def _check_features(self, features, X):
        '''
        tests output of each feature against a segmented time series X

        Parameters
        ----------
        features : dict
            feature function dictionary
        X : array-like, shape [n_samples, segment_width, n_variables]
            segmented time series (instance)

        Returns
        -------
            ftr_sizes : dict
                number of features output by each feature function
        '''
        N = X.shape[0]
        N_fts = len(features)
        fshapes = np.zeros((N_fts, 2), dtype=np.int)
        keys = [key for key in features]
        for i in np.arange(N_fts):
            fshapes[i] = features[keys[i]](X).shape

        # make sure each feature returns an array shape [N, ]
        assert np.count_nonzero(fshapes[:,0] == N)
        return {keys[i]:fshapes[i,1] for i in range(N_fts)}


    def _generate_feature_labels(self, X):
        '''
        Generates string feature labels
        '''
        ftr_sizes = self._check_features(self.features, X['ts'][0])
        f_labels = []

        # calculated features
        for key in ftr_sizes:
            for i in range(ftr_sizes[key]):
                f_labels += [key+'_'+str(i)]

        # static features
        s_labels = [s for s in X.dtype.names if s != 'ts']
        f_labels += s_labels

        return f_labels

    def _arglist(self, X):
        ''' generates argument list for threading feature computation from rec-array time-series data '''
        N = len(X)
        return [[X[i], list(self.features.values()), self.corr_features] for i in range(N)]

def _feature_thread(args):
    ''' helper function for threading '''
    return _compute_features(*args)

def _compute_features(X, features):
    '''
    Computes features for a segmented time series
    Parameters
    ----------
    X : recarray element or dict
        segmented time series data keyed ['ts']
        and any other static data associated for the series
    features : dict
        dictionary of feature methods

    Returns
    -------
        fts : array-like shape [n_samples, n_features]
        computed features

    '''
    N = X['ts'].shape[0]
    #W = X.shape[1]
    D = X['ts'].shape[2]

    Nuv = len(features)
    fts = np.zeros((N, Nuv * D))

    # computed features
    for i in range(Nuv):
        fts[:, i * D:((i + 1) * D)] = features[i](X['ts'])

    # static features
    hnames = [h for h in X.dtype.names if h != 'ts']
    for h in hnames:
        fts = np.column_stack([fts,np.full(N,X[h])])

    return fts

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
    ''' statistical mean for each variable in a segmented time series '''
    return np.mean(X, axis = 1)

def std(X):
    ''' statistical standard deviation for each variable in a segmented time series '''
    return np.std(X, axis = 1)

def var(X):
    ''' statistical variance for each variable in a segmented time series '''
    return np.std(X, axis = 1)

def min(X):
    ''' minimum value for each variable in a segmented time series '''
    return np.min(X, axis = 1)

def max(X):
    ''' maximum value for each variable in a segmented time series '''
    return np.max(X, axis = 1)

def skew(X):
    ''' skewness for each variable in a segmented time series '''
    return stats.skew(X, axis = 1)

def kurt(X):
    ''' kurtosis for each variable in a segmented time series '''
    return stats.kurtosis(X, axis = 1)

def hist4(X):
    '''
    4 bin histogram for each variable in a segmented time series
    .. note:: this feature is expensive to compute with the current implementation
    '''
    N = X.shape[0]
    D = X.shape[2]
    bins = 4
    hist = np.zeros((N, D * bins))
    for i in np.arange(N):
        for j in np.arange(D):
            hist[i,(j*D):((j+1)*D)] = np.histogram(X[i,:,j],bins = bins, density=True)[0]
    return hist

def mse(X):
    ''' computes mean spectral energy for each variable in a segmented time series '''
    return np.mean(np.square(np.abs(np.fft.fft(X, axis=1))), axis=1)

def corr2(X):
    '''
    computes correlations between all variable pairs in a segmented time series
    .. note:: this feature is expensive to compute with the current implementation
    '''
    N = X.shape[0]
    D = X.shape[2]
    trii = np.triu_indices(D, k=1)
    DD = len(trii[0])
    r = np.zeros((N, DD))
    for i in np.arange(N):
        rmat = np.corrcoef(X[i])  # get the ith window from each signal, result will be DxD
        r[i] = rmat[trii]
    return r