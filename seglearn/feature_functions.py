'''
This module has functions that can be used to compute features from a segmented time series

Sets of these functions can be passed in a dictionary object to initialize the ``SegFeatures`` transformer.

All functions follow the same template and process a single segmented time series instance:

    >>> def compute_feature(X):
    >>>     F = np.mean(X, axis = 1)
    >>>     return F
    X : array-like shape [n_samples, segment_width, n_variables]
    F : array-like [n_samples, n_features]
    The number of features returned (n_features) must be >= 1

    .. note:: ``np.atleast_3d`` is used if accessing the third dimension, as some datasets will have only a single time series variable. See ``hist4`` as an example.

Examples
--------
>>> from seglearn.feature_functions import all_features
>>> from seglearn.transform import SegFeatures
>>> FeatureTransform = SegFeatures(features=all_features())

'''

# Author: David Burns
# License: BSD

import numpy as np
from scipy import stats

def base_features():
    ''' Returns dictionary of some basic features that can be calculated for segmented time series data '''
    features = {'mean': mean,
                'std': std,
                'var': var,
                'mnx': mean_crossings}
    return features

def all_features():
    ''' Returns dictionary of all features in the module

    .. note:: Some of the features (hist4, corr) are relatively expensive to compute
    '''
    features = {'mean' : mean,
                'std': std,
                'var': var,
                'min': min,
                'max': max,
                'skew': skew,
                'kurt': kurt,
                'hist4': hist4,
                'mse': mse,
                'mnx': mean_crossings,
                'corr': corr2}
    return features

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
    ''' 4 bin histogram for each variable in a segmented time series

    .. note:: this feature is expensive to compute with the current implementation
    '''
    X = np.atleast_3d(X)
    N = X.shape[0]
    D = X.shape[2]
    bins = 4
    hist = np.zeros((N, D * bins))
    for i in np.arange(N):
        for j in np.arange(D):
            # for each variable, advance by bins
            hist[i,(j*bins):((j+1)*bins)] = np.histogram(X[i,:,j],bins = bins, density=True)[0]
    return hist

def mse(X):
    ''' computes mean spectral energy for each variable in a segmented time series '''
    return np.mean(np.square(np.abs(np.fft.fft(X, axis=1))), axis=1)

def mean_crossings(X):
    ''' Computes number of mean crossings for each variable in a segmented time series '''
    X = np.atleast_3d(X)
    N = X.shape[0]
    D = X.shape[2]
    mnx = np.zeros((N,D))
    for i in range(D):
        pos = X[:,:,i] > 0
        npos = ~pos
        c = (pos[:,:-1] & npos[:,1:]) | (npos[:,:-1] & pos[:,1:])
        mnx[:,i] = np.count_nonzero(c,axis=1)
    return mnx


def corr2(X):
    ''' computes correlations between all variable pairs in a segmented time series

    .. note:: this feature is expensive to compute with the current implementation, and cannot be used with univariate time series
    '''
    X = np.atleast_3d(X)
    N = X.shape[0]
    D = X.shape[2]
    assert D > 1
    trii = np.triu_indices(D, k=1)
    DD = len(trii[0])
    r = np.zeros((N, DD))
    for i in np.arange(N):
        rmat = np.corrcoef(X[i])  # get the ith window from each signal, result will be DxD
        r[i] = rmat[trii]
    return r