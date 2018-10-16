'''
This module is for transforming time series data.
'''
# Author: David Burns
# License: BSD

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.exceptions import NotFittedError
from scipy.interpolate import interp1d

from .feature_functions import base_features
from .base import TS_Data
from .util import get_ts_data_parts, check_ts_data

__all__ = ['SegmentX', 'SegmentXY', 'SegmentXYForecast', 'PadTrunc', 'Interp', 'FeatureRep']


class XyTransformerMixin(object):
    ''' Base class for transformer that transforms data and target '''

    def fit_transform(self, X, y, sample_weight=None, **fit_params):
        '''
        Fit the data and transform (required by sklearn API)

        Parameters
        ----------
        X : array-like, shape [n_series, ...]
           Time series data and (optionally) contextual data
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
        return self.fit(X, y, **fit_params).transform(X, y, sample_weight)


def last(y):
    ''' Returns the last column from 2d matrix '''
    return y[:, (y.shape[1] - 1)]


def middle(y):
    ''' Returns the middle column from 2d matrix '''
    return y[:, y.shape[1] // 2]


def mean(y):
    ''' returns average along axis 1'''
    return np.mean(y, axis=1)


def every(y):
    ''' Returns all values (sequences) of y '''
    return y


def shuffle_data(X, y=None, sample_weight=None):
    ''' Shuffles indices X, y, and sample_weight together'''
    if len(X) > 1:
        ind = np.arange(len(X), dtype=np.int)
        np.random.shuffle(ind)
        Xt = X[ind]
        yt = y
        swt = sample_weight

        if yt is not None:
            yt = yt[ind]
        if swt is not None:
            swt = swt[ind]

        return Xt, yt, swt
    else:
        return X, y, sample_weight


class SegmentX(BaseEstimator, XyTransformerMixin):
    '''
    Transformer for sliding window segmentation for datasets where
    X is time series data, optionally with contextual variables
    and each time series in X has a single target value y

    The target y is mapped to all segments from their parent series.
    The transformed data consists of segment/target pairs that can be learned
    through a feature representation or directly with a neural network.

    Parameters
    ----------
    width : int > 0
        width of segments (number of samples)
    overlap : float range [0,1)
        amount of overlap between segments. must be in range: 0 <= overlap < 1.
        shuffle : bool, optional
        shuffle the segments before fitting the ``est`` pipeline (recommended)
    shuffle : bool, optional
        shuffle the segments after transform (recommended for batch optimizations)
    random_state : int, default = None
        Randomized segment shuffling will return different results for each call to
        ``transform``. If you have set ``shuffle`` to True and want the same result
        with each call to ``fit``, set ``random_state`` to an integer.

    Todo
    ----
    separate fit and predict overlap parameters
    '''

    def __init__(self, width=100, overlap=0.5, shuffle=False, random_state=None):
        self.width = width
        self.overlap = overlap
        self.shuffle = shuffle
        self.random_state = random_state
        self._validate_params()

        self.f_labels = None
        self.step = int(self.width * (1. - self.overlap))
        self.step = self.step if self.step >= 1 else 1

    def _validate_params(self):
        if not self.width >= 1:
            raise ValueError("width must be >=1 (was %d)" % self.width)
        if not (self.overlap >= 0.0 and self.overlap <= 1.0):
            raise ValueError("overlap must be >=0 and <=1.0 (was %.2f)" % self.overlap)

    def fit(self, X, y=None):
        '''
        Fit the transform

        Parameters
        ----------
        X : array-like, shape [n_series, ...]
            Time series data and (optionally) contextual data
        y : None
            There is no need of a target in a transformer, yet the pipeline API requires
            this parameter.
        shuffle : bool
            Shuffles data after transformation

        Returns
        -------
        self : object
            Returns self.
        '''
        check_ts_data(X, y)
        return self

    def transform(self, X, y=None, sample_weight=None):
        '''
        Transforms the time series data into segments (temporal tensor)
        Note this transformation changes the number of samples in the data
        If y and sample_weight are provided, they are transformed to align to the new samples


        Parameters
        ----------
        X : array-like, shape [n_series, ...]
           Time series data and (optionally) contextual data
        y : array-like shape [n_series], default = None
            target vector
        sample_weight : array-like shape [n_series], default = None
            sample weights

        Returns
        -------
        Xt : array-like, shape [n_segments, ]
            transformed time series data
        yt : array-like, shape [n_segments]
            expanded target vector
        sample_weight_new : array-like shape [n_segments]
            expanded sample weights
        '''
        check_ts_data(X, y)
        Xt, Xc = get_ts_data_parts(X)
        yt = y
        swt = sample_weight

        N = len(Xt)  # number of time series

        if Xt[0].ndim > 1:
            Xt = np.array([sliding_tensor(Xt[i], self.width, self.step) for i in np.arange(N)])
        else:
            Xt = np.array([sliding_window(Xt[i], self.width, self.step) for i in np.arange(N)])

        Nt = [len(Xt[i]) for i in np.arange(len(Xt))]
        Xt = np.concatenate(Xt)

        if yt is not None:
            yt = expand_variables_to_segments(yt, Nt).ravel()

        if swt is not None:
            swt = expand_variables_to_segments(swt, Nt).ravel()

        if Xc is not None:
            Xc = expand_variables_to_segments(Xc, Nt)
            Xt = TS_Data(Xt, Xc)

        if self.shuffle is True:
            check_random_state(self.random_state)
            return shuffle_data(Xt, yt, swt)

        return Xt, yt, swt


class SegmentXY(BaseEstimator, XyTransformerMixin):
    '''
    Transformer for sliding window segmentation for datasets where
    X is time series data, optionally with contextual variables
    and y is also time series data with the same sampling interval as X

    The target y is mapped to segments from their parent series,
    using the parameter ``y_func`` to determine the mapping behavior.
    The segment targets can be a single value, or a sequence of values
    depending on ``y_func`` parameter.

    The transformed data consists of segment/target pairs that can be learned
    through a feature representation or directly with a neural network.


    Parameters
    ----------
    width : int > 0
        width of segments (number of samples)
    overlap : float range [0,1)
        amount of overlap between segments. must be in range: 0 <= overlap < 1.
    y_func : function
        returns target from array of target segments (eg ``last``, ``middle``, or ``mean``)
    shuffle : bool, optional
        shuffle the segments after transform (recommended for batch optimizations)
    random_state : int, default = None
        Randomized segment shuffling will return different results for each call to ``transform``.
        If you have set ``shuffle`` to True and want the same result with each call to ``fit``,
        set ``random_state`` to an integer.

    Returns
    -------
    self : object
        Returns self.
    '''

    def __init__(self, width=100, overlap=0.5, y_func=last, shuffle=False, random_state=None):
        self.width = width
        self.overlap = overlap
        self.y_func = y_func
        self.shuffle = shuffle
        self.random_state = random_state
        self._validate_params()

        self.step = int(self.width * (1. - self.overlap))
        self.step = self.step if self.step >= 1 else 1

    def _validate_params(self):
        if not self.width >= 1:
            raise ValueError("width must be >=1 (was %d)" % self.width)
        if not (self.overlap >= 0.0 and self.overlap <= 1.0):
            raise ValueError("overlap must be >=0 and <=1.0 (was %.2f)" % self.overlap)

    def fit(self, X, y=None):
        '''
        Fit the transform

        Parameters
        ----------
        X : array-like, shape [n_series, ...]
            Time series data and (optionally) contextual data
        y : None
            There is no need of a target in a transformer, yet the pipeline API requires this
            parameter.

        Returnsself._validate_params()
        -------
        self : object
            Returns self.
        '''
        check_ts_data(X, y)
        return self

    def transform(self, X, y=None, sample_weight=None):
        '''
        Transforms the time series data into segments
        Note this transformation changes the number of samples in the data
        If y is provided, it is segmented and transformed to align to the new samples as per
        ``y_func``
        Currently sample weights always returned as None

        Parameters
        ----------
        X : array-like, shape [n_series, ...]
           Time series data and (optionally) contextual data
        y : array-like shape [n_series], default = None
            target vector
        sample_weight : array-like shape [n_series], default = None
            sample weights

        Returns
        -------
        Xt : array-like, shape [n_segments, ]
            transformed time series data
        yt : array-like, shape [n_segments]
            expanded target vector
        sample_weight_new : None

        '''
        check_ts_data(X, y)
        Xt, Xc = get_ts_data_parts(X)
        yt = y

        N = len(Xt)  # number of time series

        if Xt[0].ndim > 1:
            Xt = np.array([sliding_tensor(Xt[i], self.width, self.step) for i in np.arange(N)])
        else:
            Xt = np.array([sliding_window(Xt[i], self.width, self.step) for i in np.arange(N)])

        Nt = [len(Xt[i]) for i in np.arange(len(Xt))]
        Xt = np.concatenate(Xt)

        if Xc is not None:
            Xc = expand_variables_to_segments(Xc, Nt)
            Xt = TS_Data(Xt, Xc)

        if yt is not None:
            yt = np.array([sliding_window(yt[i], self.width, self.step) for i in np.arange(N)])
            yt = np.concatenate(yt)
            yt = self.y_func(yt)

        if self.shuffle is True:
            check_random_state(self.random_state)
            Xt, yt, _ = shuffle_data(Xt, yt)

        return Xt, yt, None


class SegmentXYForecast(BaseEstimator, XyTransformerMixin):
    '''
    Forecast sliding window segmentation for time series or sequence datasets

    The target y is mapped to segments from their parent series,
    using the ``forecast`` and ``y_func`` parameters to determine the mapping behavior.
    The segment targets can be a single value, or a sequence of values
    depending on ``y_func`` parameter.

    The transformed data consists of segment/target pairs that can be learned
    through a feature representation or directly with a neural network.

    Parameters
    ----------
    width : int > 0
        width of segments (number of samples)
    overlap : float range [0,1)
        amount of overlap between segments. must be in range: 0 <= overlap < 1.
    forecast : int
        The number of samples ahead in time to forecast
    y_func : function
        returns target from array of target forecast segments (eg ``last``, or ``mean``)
    shuffle : bool, optional
        shuffle the segments after transform (recommended for batch optimizations)
    random_state : int, default = None
        Randomized segment shuffling will return different results for each call to ``transform``.
        If you have set ``shuffle`` to True and want the same result with each call to ``fit``, set
        ``random_state`` to an integer.

    Returns
    -------
    self : object
        Returns self.
    '''

    def __init__(self, width=100, overlap=0.5, forecast=10, y_func=last, shuffle=False,
                 random_state=None):
        self.width = width
        self.overlap = overlap
        self.forecast = forecast
        self.y_func = y_func
        self.shuffle = shuffle
        self.random_state = random_state
        self._validate_params()

        self.step = int(self.width * (1. - self.overlap))
        self.step = self.step if self.step >= 1 else 1

    def _validate_params(self):
        if not self.width >= 1:
            raise ValueError("width must be >=1 (was %d)" % self.width)
        if not (self.overlap >= 0.0 and self.overlap <= 1.0):
            raise ValueError("overlap must be >=0 and <=1.0 (was %.2f)" % self.overlap)
        if not self.forecast >= 1:
            raise ValueError("forecase must be >=1 (was %d)" % self.forecast)

    def fit(self, X=None, y=None):
        '''
        Fit the transform

        Parameters
        ----------
        X : array-like, shape [n_series, ...]
            Time series data and (optionally) contextual data
        y : None
            There is no need of a target in a transformer, yet the pipeline API requires this
            parameter.

        Returns
        -------
        self : object
            Returns self.
        '''
        check_ts_data(X, y)
        return self

    def transform(self, X, y, sample_weight=None):
        '''
        Forecast sliding window segmentation for time series or sequence datasets.
        Note this transformation changes the number of samples in the data.
        Currently sample weights always returned as None.

        Parameters
        ----------
        X : array-like, shape [n_series, ...]
           Time series data and (optionally) contextual data
        y : array-like shape [n_series]
            target vector
        sample_weight : array-like shape [n_series], default = None
            sample weights

        Returns
        -------
        X_new : array-like, shape [n_segments, ]
            segmented X data
        y_new : array-like, shape [n_segments]
            forecast y data
        sample_weight_new : None

        '''
        check_ts_data(X, y)
        Xt, Xc = get_ts_data_parts(X)
        yt = y

        # if only one time series is learned
        if len(Xt[0]) == 1:
            Xt = [Xt]

        N = len(Xt)  # number of time series

        if Xt[0].ndim > 1:
            Xt = np.array([sliding_tensor(Xt[i], self.width + self.forecast, self.step) for i in
                           np.arange(N)])
        else:
            Xt = np.array([sliding_window(Xt[i], self.width + self.forecast, self.step) for i in
                           np.arange(N)])

        Nt = [len(Xt[i]) for i in np.arange(len(Xt))]
        Xt = np.concatenate(Xt)

        # todo: implement advance X
        Xt = Xt[:, 0:self.width]

        if Xc is not None:
            Xc = expand_variables_to_segments(Xc, Nt)
            Xt = TS_Data(Xt, Xc)

        if yt is not None:
            yt = np.array([sliding_window(yt[i], self.width + self.forecast, self.step) for i in
                           np.arange(N)])
            yt = np.concatenate(yt)
            yt = yt[:, self.width:(self.width + self.forecast)]  # target y
            yt = self.y_func(yt)

        if self.shuffle is True:
            check_random_state(self.random_state)
            Xt, yt, _ = shuffle_data(Xt, yt)

        return Xt, yt, None


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
    return w.reshape((int(len(w) / width), width), order='F')


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
    return np.stack(data, axis=2)


class PadTrunc(BaseEstimator, XyTransformerMixin):
    '''
    Transformer for using padding and truncation to enforce fixed length on all time
    series in the dataset. Series' longer than ``width`` are truncated to length ``width``.
    Series' shorter than length ``width`` are padded at the end with zeros up to length ``width``.

    The same behavior is applied to the target if it is a series and passed to the transformer.

    Parameters
    ----------
    width : int >= 1
        width of segments (number of samples)
    '''

    def __init__(self, width=100):
        if not width >= 1:
            raise ValueError("width must be >= 1 (was %d)" % width)
        self.width = width

    def _mv_resize(self, v):
        N = len(v)
        if v[0].ndim > 1:
            D = v[0].shape[1]
            w = np.zeros((N, self.width, D))
        else:
            w = np.zeros((N, self.width))
        for i in np.arange(N):
            Ni = min(self.width, len(v[i]))
            w[i, 0:Ni] = v[i][0:Ni]
        return w

    def fit(self, X, y=None):
        '''
        Fit the transform. Does nothing, for compatibility with sklearn API.

        Parameters
        ----------
        X : array-like, shape [n_series, ...]
            Time series data and (optionally) contextual data
        y : None
            There is no need of a target in a transformer, yet the pipeline API requires this
            parameter.

        Returns
        -------
        self : object
            Returns self.
        '''
        check_ts_data(X, y)
        return self

    def transform(self, X, y=None, sample_weight=None):
        '''
        Transforms the time series data into fixed length segments using padding and or truncation
        If y is a time series and passed, it will be transformed as well

        Parameters
        ----------
        X : array-like, shape [n_series, ...]
           Time series data and (optionally) contextual data
        y : array-like shape [n_series], default = None
            target vector
        sample_weight : array-like shape [n_series], default = None
            sample weights

        Returns
        -------
        X_new : array-like, shape [n_series, ]
            transformed time series data
        y_new : array-like, shape [n_series]
            expanded target vector
        sample_weight_new : None

        '''
        check_ts_data(X, y)
        Xt, Xc = get_ts_data_parts(X)
        yt = y
        swt = sample_weight

        Xt = self._mv_resize(Xt)

        if Xc is not None:
            Xt = TS_Data(Xt, Xc)

        if yt is not None and len(np.atleast_1d(yt[0])) > 1:
            # y is a time series
            yt = self._mv_resize(yt)
            swt = None
        elif yt is not None:
            # todo: is this needed?
            yt = np.array(yt)

        return Xt, yt, swt


class Interp(BaseEstimator, XyTransformerMixin):
    '''
    Transformer for resampling time series data to a fixed period over closed interval
    (direct value interpolation).
    Default interpolation is linear, but other types can be specified.
    If the target is a series, it will be resampled as well.

    categorical_target should be set to True if the target series is a class
    The transformer will then use nearest neighbor interp on the target.

    This transformer assumes the time dimension is column 0, i.e. X[0][:,0]
    Note the time dimension is removed, since this becomes a linear sequence.
    If start time or similar is important to the estimator, use a context variable.

    Parameters
    ----------
    sample_period : numeric
        desired sampling period
    kind : string
        interpolation type - valid types as per scipy.interpolate.interp1d
    categorical_target : bool
        set to True for classification problems nearest use nearest instead of linear interp

    '''

    def __init__(self, sample_period, kind='linear', categorical_target=False):
        if not sample_period > 0:
            raise ValueError("sample_period must be >0 (was %f)" % sample_period)

        self.sample_period = sample_period
        self.kind = kind
        self.categorical_target = categorical_target

    def fit(self, X, y=None):
        '''
        Fit the transform. Does nothing, for compatibility with sklearn API.

        Parameters
        ----------
        X : array-like, shape [n_series, ...]
            Time series data and (optionally) contextual data
        y : None
            There is no need of a target in a transformer, yet the pipeline API requires this
            parameter.

        Returns
        -------
        self : object
            Returns self.
        '''
        check_ts_data(X, y)
        if not X[0].ndim > 1:
            raise ValueError("X variable must have more than 1 channel")

        return self

    def _interp(self, t_new, t, x, kind):
        interpolator = interp1d(t, x, kind=kind, copy=False, bounds_error=False,
                                fill_value="extrapolate", assume_sorted=True)
        return interpolator(t_new)

    def transform(self, X, y=None, sample_weight=None):
        '''
        Transforms the time series data with linear direct value interpolation
        If y is a time series and passed, it will be transformed as well
        The time dimension is removed from the data

        Parameters
        ----------
        X : array-like, shape [n_series, ...]
           Time series data and (optionally) contextual data
        y : array-like shape [n_series], default = None
            target vector
        sample_weight : array-like shape [n_series], default = None
            sample weights

        Returns
        -------
        X_new : array-like, shape [n_series, ]
            transformed time series data
        y_new : array-like, shape [n_series]
            expanded target vector
        sample_weight_new : array-like or None
            None is returned if target is changed. Otherwise it is returned unchanged.
        '''
        check_ts_data(X, y)
        Xt, Xc = get_ts_data_parts(X)
        yt = y
        swt = sample_weight

        N = len(Xt)  # number of series
        D = Xt[0].shape[1] - 1  # number of data channels

        # 1st channel is time
        t = [Xt[i][:, 0] for i in np.arange(N)]
        t_lin = [np.arange(Xt[i][0, 0], Xt[i][-1, 0], self.sample_period) for i in np.arange(N)]

        if D == 1:
            Xt = [self._interp(t_lin[i], t[i], Xt[i][:, 1], kind=self.kind) for i in np.arange(N)]
        elif D > 1:
            Xt = [np.column_stack([self._interp(t_lin[i], t[i], Xt[i][:, j], kind=self.kind)
                                   for j in range(1, D)]) for i in np.arange(N)]
        if Xc is not None:
            Xt = TS_Data(Xt, Xc)

        if yt is not None and len(np.atleast_1d(yt[0])) > 1:
            # y is a time series
            swt = None
            if self.categorical_target is True:
                yt = [self._interp(t_lin[i], t[i], yt[i], kind='nearest') for i in np.arange(N)]
            else:
                yt = [self._interp(t_lin[i], t[i], yt[i], kind=self.kind) for i in np.arange(N)]
        else:
            # y is static - leave y alone
            pass

        return Xt, yt, swt


class FeatureRep(BaseEstimator, TransformerMixin):
    '''
    A transformer for calculating a feature representation from segmented time series data.

    This transformer calculates features from the segmented time series', by computing the same
    feature set for each segment from each time series in the data set.

    The ``features`` computed are a parameter of this transformer, defined by a dict of functions.
    The seglearn package includes some useful features, but this basic feature set can be easily
    extended.

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

        If features is not specified, a default feature dictionary will be used (see base_features).
        See ``feature_functions`` for example implementations.

    Attributes
    ----------
    f_labels : list of string feature labels (in order) corresponding to the computed features

    Examples
    --------

    >>> from seglearn.transform import FeatureRep, SegmentX
    >>> from seglearn.pipe import Pype
    >>> from seglearn.feature_functions import mean, var, std, skew
    >>> from seglearn.datasets import load_watch
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> data = load_watch()
    >>> X = data['X']
    >>> y = data['y']
    >>> fts = {'mean': mean, 'var': var, 'std': std, 'skew': skew}
    >>> clf = Pype([('seg', SegmentX()),
    >>>             ('ftr', FeatureRep(features = fts))),
    >>>             ('rf',RandomForestClassifier())])
    >>> clf.fit(X, y)
    >>> print(clf.score(X, y))

    '''

    def __init__(self, features='default'):
        if features == 'default':
            self.features = base_features()
        else:
            if not isinstance(features, dict):
                raise TypeError("features must either 'default' or an instance of type dict")
            self.features = features

        self.f_labels = None

    def fit(self, X, y=None):
        '''
        Fit the transform

        Parameters
        ----------
        X : array-like, shape [n_series, ...]
            Segmented time series data and (optionally) contextual data
        y : None
            There is no need of a target in a transformer, yet the pipeline API requires this
            parameter.

        Returns
        -------
        self : object
            Returns self.
        '''
        check_ts_data(X, y)
        self._reset()
        print("X Shape: ", X.shape)
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
        self._check_if_fitted()
        Xt, Xc = get_ts_data_parts(X)
        check_array(Xt, dtype='numeric', ensure_2d=False, allow_nd=True)

        fts = np.column_stack([self.features[f](Xt) for f in self.features])
        if Xc is not None:
            fts = np.column_stack([fts, Xc])
        return fts

    def _reset(self):
        ''' Resets internal data-dependent state of the transformer. __init__ parameters not
        touched. '''
        self.f_labels = None

    def _check_if_fitted(self):
        if self.f_labels is None:
            raise NotFittedError("FeatureRep")

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
        if not np.all(fshapes[:, 0] == N):
            raise ValueError("feature function returned array with invalid length, ",
                             np.array(features.keys())[fshapes[:, 0] != N])

        return {keys[i]: fshapes[i, 1] for i in range(N_fts)}

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
                f_labels += [key + '_' + str(i)]

        # contextual features
        if Xc is not None:
            Ns = len(np.atleast_1d(Xc[0]))
            s_labels = ["context_" + str(i) for i in range(Ns)]
            f_labels += s_labels

        return f_labels
