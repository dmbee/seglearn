'''
This module is for transforming time series data.
'''
# Author: David Burns
# License: BSD

from .util import make_ts_data, get_ts_data_parts
from .feature_functions import base_features

import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


__all__ = ['Segment','SegFeatures']


class Segment(BaseEstimator, TransformerMixin):
    '''
    Transformer for sliding window segmentation of a time series or sequence

    Parameters
    ----------
    width : int > 0
        width of segments (number of samples)
    overlap : float range [0,1)
        amount of overlap between segments. must be in range: 0 <= overlap < 1.

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
        Fit the transform

        Parameters
        ----------
        X : array-like, shape [n_series, ...]
            Time series data and (optionally) static data created as per ``make_ts_data``
        y : None
            There is no need of a target in a transformer, yet the pipeline API requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        '''
        self._reset()
        assert self.width > 0
        assert self.overlap >= 0. and self.overlap <= 1
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
        ----------
        X : array-like, shape [n_series, ...]
           Time series data and (optionally) static data created as per ``make_ts_data``

        Returns
        -------
        X_new : array-like shape [n_series, ...]
            Segmented time series data and (optionally) static data

        .. note:: The input data ``X`` must have the time series data in column 0. Each element of ``X[:,0]`` will have shape [n_samples, n_variables]. Each element of ``X_new[:,0]`` will have shape [n_segments, width, n_variables].

        '''
        check_is_fitted(self, 'step')
        Xt, Xs = get_ts_data_parts(X)
        N = len(Xt)

        if Xt[0].ndim > 1:
            Xt = np.array([sliding_tensor(Xt[i], self.width, self.step) for i in np.arange(N)])
        else:
            Xt = np.array([sliding_window(Xt[i], self.width, self.step) for i in np.arange(N)])

        return make_ts_data(Xt, Xs)

    def fit_transform(self, X, y = None):
        self.fit(X, y)
        return self.transform(X)

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


class SegFeatures(BaseEstimator, TransformerMixin):
    '''
    A transformer for calculating a feature representation from segmented time series data.

    This transformer generates tabular feature data from the segmented time series', by computing the same feature set for each segment from each time series in the data set. This transformer, if used, follows the Segment transformer in the ``feed`` pipeline for the ``SegPipe`` class.

    The input data ``X`` contains segmented time series data, where each elements have shape [n_segments, width, n_variables]. n_segments is different for each element. If ``X`` also contains static variables, the segmented time series data must be in column 0. The transform method of this class generates a feature representation ``X_new``, where each element has shape [n_segments, n_features] and includes the computed features and static data.


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

        If features is not specified, a default feature dictionary will be used (see base_features). See ``feature_functions`` for example implementations.

    multithread: bool, optional
        If true, will use multithreading to compute the features

    Attributes
    ----------
    f_labels : list of string feature labels (in order) corresponding to the computed features

    Examples
    --------

    >>> from seglearn.transform import SegFeatures, Segment
    >>> from seglearn.feature_functions import mean, var, std, skew
    >>> from seglearn.pipe import SegPipe
    >>> from seglearn.datasets import load_watch
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> data = load_watch()
    >>> fts = {'mean': mean, 'var': var, 'std': std, 'skew': skew}
    >>> feed = Pipeline([('segment', Segment()),('features', SegFeatures(fts))])
    >>> est = RandomForestClassifier()
    >>> pipe = SegPipe(feed, est)
    >>> pipe.fit(data['X'], data['y'])
    >>> print(pipe.score(data['X'], data['y']))

    '''
    def __init__(self, features = base_features(), multithread = False):
        self.multithread = multithread
        self.features = features

    def fit(self, X, y = None):
        '''
        Fit the transform

        Parameters
        ----------
        X : array-like, shape [n_series, ...]
            Segmented time series data and (optionally) static data
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
        ----------
        X : array-like, shape [n_series, ...]
            Segmented time series data and (optionally) static data

        Returns
        -------
        X_new : array shape [n_series, ...]
            Feature representation of segmented time series data and static data

        '''
        check_is_fitted(self,'f_labels')
        Xt, Xs = get_ts_data_parts(X)
        N = len(Xt)

        if Xs is not None:
            arglist = [[Xt[i], Xs[i], list(self.features.values())] for i in range(N)]
        else:
            arglist = [[Xt[i], None, list(self.features.values())] for i in range(N)]

        if self.multithread == True:
            pool = Pool(cpu_count())
            X_new = pool.map(_feature_thread, arglist)
        else:
            X_new = []
            for i in range(N):
                X_new.append(_feature_thread(arglist[i]))


        return make_ts_data(np.array(X_new))

    def _reset(self):
        ''' Resets internal data-dependent state of the transformer. __init__ parameters not touched. '''
        if hasattr(self, 'f_labels'):
            del self.f_labels

    def _check_features(self, features, Xti):
        '''
        tests output of each feature against a segmented time series X

        Parameters
        ----------
        features : dict
            feature function dictionary
        Xti : array-like, shape [n_samples, segment_width, n_variables]
            segmented time series (instance)

        Returns
        -------
            ftr_sizes : dict
                number of features output by each feature function
        '''
        N = Xti.shape[0]
        N_fts = len(features)
        fshapes = np.zeros((N_fts, 2), dtype=np.int)
        keys = [key for key in features]
        for i in np.arange(N_fts):
            fshapes[i] = np.row_stack(features[keys[i]](Xti)).shape

        # make sure each feature returns an array shape [N, ]
        assert np.count_nonzero(fshapes[:,0] == N)
        return {keys[i]:fshapes[i,1] for i in range(N_fts)}


    def _generate_feature_labels(self, X):
        '''
        Generates string feature labels
        '''
        Xt, Xs = get_ts_data_parts(X)

        ftr_sizes = self._check_features(self.features, Xt[0])
        f_labels = []

        # calculated features
        for key in ftr_sizes:
            for i in range(ftr_sizes[key]):
                f_labels += [key+'_'+str(i)]

        # static features
        if Xs is not None:
            Ns = len(np.atleast_1d(Xs[0]))
            s_labels = ["static_"+str(i) for i in range(Ns)]
            f_labels += s_labels

        return f_labels

def _feature_thread(args):
    ''' helper function for threading '''
    return _compute_features(*args)

def _compute_features(Xti, Xsi, features):
    '''
    Computes features for a segmented time series instance

    Parameters
    ----------
    Xti : array-like shape [n_segments, width, n_variables]
        segmented time series instance
    Xsi : array-like [n_static_variables]
        static variables associated with time series instance
    features :
        feature function dictionary

    Returns
    -------
    fts : array-like shape [n_segments, n_features]
        feature representation of Xti and Xsi
    '''
    N = Xti.shape[0]
    # computed features
    fts = [features[i](Xti) for i in range(len(features))]
    # static features
    s_fts = []
    if Xsi is not None:
        Ns = len(np.atleast_1d(Xsi))
        s_fts = [np.full((N,Ns), Xsi)]
    fts = np.column_stack(fts+s_fts)

    return fts

