'''
This module is for transforming time series data.
'''
# Author: David Burns
# License: BSD

from .util import make_ts_data, get_ts_data_parts
from .feature_functions import base_features

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


__all__ = ['SegmentX', 'SegmentXY', 'FeatureRep']


class XyTransformerMixin(object):
    ''' Base class for transformer that transforms data and target '''

    def fit_transform(self, X, y, sample_weight = None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X, y, sample_weight)

def last(y):
    ''' Returns the last column from 2d matrix '''
    return y[:, (y.shape[1] - 1)]

def middle(y):
    ''' Returns the middle column from 2d matrix '''
    return y[:, y.shape[1]//2]

def mean(y):
    ''' returns average along axis 1'''
    return np.mean(y, axis=1)


class SegmentX(BaseEstimator, XyTransformerMixin):
    '''
    Transformer for sliding window segmentation for datasets where
    X is time series data, optionally with contextual variables
    and each time series in X has a single target value y

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
            Time series data and (optionally) contextual data created as per ``make_ts_data``
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

    def transform(self, X, y = None, sample_weight = None):
        '''
        Transforms the time series data into segments (temporal tensor)
        Note this transformation changes the number of samples in the data
        If y and sample_weight are provided, they are transformed to align to the new samples


        Parameters
        ----------
        X : array-like, shape [n_series, ...]
           Time series data and (optionally) contextual data created as per ``make_ts_data``
        y : array-like shape [n_series], default = None
            target vector
        sample_weight : array-like shape [n_series], default = None
            sample weights

        Returns
        -------
        X_new : array-like, shape [n_segments, ]
            transformed time series data
        y_new : array-like, shape [n_segments]
            expanded target vector
        sample_weight_new : array-like shape [n_segments]
            expanded sample weights
        '''
        check_is_fitted(self, 'step')
        Xt, Xc = get_ts_data_parts(X)

        N = len(Xt)  # number of time series

        if Xt[0].ndim > 1:
            Xt = np.array([sliding_tensor(Xt[i], self.width, self.step) for i in np.arange(N)])
        else:
            Xt = np.array([sliding_window(Xt[i], self.width, self.step) for i in np.arange(N)])

        Nt = [len(Xt[i]) for i in np.arange(len(Xt))]
        Xt = np.concatenate(Xt)

        if y is not None:
            y = self._expand_target_to_segments(y, Nt)

        if sample_weight is not None:
            sample_weight = self._expand_target_to_segments(sample_weight, Nt)

        if Xc is None:
            return Xt, y, sample_weight
        else:
            Xc = expand_variables_to_segments(Xc, Nt)
            X = make_ts_data(Xt, Xc)
            return X, y, sample_weight


    def _expand_target_to_segments(self, y, Nt):
        ''' expands variable vector v, by repeating each instance as specified in Nt '''
        y_e = np.concatenate([np.full(Nt[i], y[i]) for i in np.arange(len(y))])
        return y_e



class SegmentXY(BaseEstimator, XyTransformerMixin):
    '''
    Transformer for sliding window segmentation for datasets where
    X is time series data, optionally with contextual variables
    and y is also time series data with the same sampling interval as X


    Parameters
    ----------
    X : array-like, shape [n_series, ...]
        Time series data and (optionally) contextual data created as per ``make_ts_data``
    y : None
        There is no need of a target in a transformer, yet the pipeline API requires this parameter.
    y_func : function
        returns target from array of target segments (eg ``last``, ``middle``, or ``mean``)
    forecast : int, default = None
        if set, will shift target by this number of segments forward in time

    Returns
    -------
    self : object
        Returns self.
    '''

    def __init__(self, width = 100, overlap = 0.5, y_func = last, forecast = None):
        self.width = width
        self.overlap = overlap
        self.y_func = y_func
        self.forecast = forecast

    def fit(self, X, y = None):
        '''
        Fit the transform

        Parameters
        ----------
        X : array-like, shape [n_series, ...]
            Time series data and (optionally) contextual data created as per ``make_ts_data``
        y : None
            There is no need of a target in a transformer, yet the pipeline API requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        '''
        self._reset()
        self.step = int(self.width * (1. - self.overlap))
        self.step = self.step if self.step >= 1 else 1
        return self

    def _reset(self):
        ''' Resets internal data-dependent state of the transformer. __init__ parameters not touched. '''
        if hasattr(self, 'step'):
            del self.step

    def _validate_params(self):
        assert self.width > 0
        assert self.overlap >= 0. and self.overlap <= 1.
        if self.forecast is not None:
            assert self.forecast > 0

    def transform(self, X, y = None, sample_weight = None):
        '''
        Transforms the time series data into segments
        Note this transformation changes the number of samples in the data
        If y is provided, it is segmented and transformed to align to the new samples as per ``y_func``
        Currently sample weights always returned as None

        Parameters
        ----------
        X : array-like, shape [n_series, ...]
           Time series data and (optionally) contextual data created as per ``make_ts_data``
        y : array-like shape [n_series], default = None
            target vector
        sample_weight : array-like shape [n_series], default = None
            sample weights

        Returns
        -------
        X_new : array-like, shape [n_segments, ]
            transformed time series data
        y_new : array-like, shape [n_segments]
            expanded target vector
        sample_weight_new : None

        '''
        check_is_fitted(self, 'step')
        Xt, Xc = get_ts_data_parts(X)

        # if only one time series is learned
        if len(Xt[0]) == 1:
            Xt = [Xt]

        N = len(Xt) # number of time series

        if Xt[0].ndim > 1:
            Xt = np.array([sliding_tensor(Xt[i], self.width, self.step) for i in np.arange(N)])
        else:
            Xt = np.array([sliding_window(Xt[i], self.width, self.step) for i in np.arange(N)])

        Nt = [len(Xt[i]) for i in np.arange(len(Xt))]
        Xt = np.concatenate(Xt)

        if Xc is None:
            X = Xt
        else:
            Xc = expand_variables_to_segments(Xc, Nt)
            X = make_ts_data(Xt, Xc)

        if y is not None:
            y = np.array([sliding_window(y[i], self.width, self.step) for i in np.arange(N)])
            y = np.concatenate(y)

            if self.forecast is not None:
                X = X[0:(len(X) - self.forecast)]
                y = y[self.forecast:len(y)]

            y = self.y_func(y)

        return X, y, None


def expand_variables_to_segments(v, Nt):
    ''' expands contextual variables v, by repeating each instance as specified in Nt '''
    N_v = len(np.atleast_1d(v[0]))
    return np.concatenate([np.full((Nt[i], N_v), v[i]) for i in np.arange(len(v))])

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


class FeatureRep(BaseEstimator, TransformerMixin):
    '''
    A transformer for calculating a feature representation from segmented time series data.

    This transformer generates tabular feature data from the segmented time series', by computing the same feature set for each segment from each time series in the data set. This transformer, if used, follows the Segment transformer in the ``feed`` pipeline for the ``SegPipe`` class.

    The input data ``X`` contains segmented time series data, where each elements have shape [n_segments, width, n_variables]. n_segments is different for each element. If ``X`` also contains contextual variables, the segmented time series data must be in column 0. The transform method of this class generates a feature representation ``X_new``, where each element has shape [n_segments, n_features] and includes the computed features and contextual data.


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

    >>> from seglearn.transform import FeatureRep
    >>> from seglearn.feature_functions import mean, var, std, skew
    >>> from seglearn.pipe import SegPipe
    >>> from seglearn.datasets import load_watch
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> data = load_watch()
    >>> X = data['X']
    >>> y = data['y']
    >>> fts = {'mean': mean, 'var': var, 'std': std, 'skew': skew}
    >>> est = Pipeline([('ftr', FeatureRep(features = fts)),('rf',RandomForestClassifier())])
    >>> pipe = SegPipe(est)
    >>> pipe.fit(X, y)
    >>> print(pipe.score(X, y))

    '''
    def __init__(self, features = base_features()):
        self.features = features

    def fit(self, X, y = None):
        '''
        Fit the transform

        Parameters
        ----------
        X : array-like, shape [n_series, ...]
            Segmented time series data and (optionally) contextual data
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
        If contextual data is included in X, it is returned with the feature data.

        Parameters
        ----------
        X : array-like, shape [n_series, ...]
            Segmented time series data and (optionally) contextual data

        Returns
        -------
        X_new : array shape [n_series, ...]
            Feature representation of segmented time series data and contextual data

        '''
        check_is_fitted(self,'f_labels')
        Xt, Xc = get_ts_data_parts(X)
        fts = np.column_stack([self.features[f](Xt) for f in self.features])
        if Xc is not None:
            fts = np.column_stack([fts,Xc])
        return fts

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
        Xt, Xc = get_ts_data_parts(X)

        ftr_sizes = self._check_features(self.features, Xt[0:3])
        f_labels = []

        # calculated features
        for key in ftr_sizes:
            for i in range(ftr_sizes[key]):
                f_labels += [key+'_'+str(i)]

        # contextual features
        if Xc is not None:
            Ns = len(np.atleast_1d(Xc[0]))
            s_labels = ["context_"+str(i) for i in range(Ns)]
            f_labels += s_labels

        return f_labels


