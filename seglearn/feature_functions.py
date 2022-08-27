"""
This module has functions or callable objects that can be used to compute features from segmented
time series data

Sets of these functions or callables can be passed in a dictionary object to initialize the
``FeatureRep`` transformer.

All functions follow the same template and process a single segmented time series instance:

    >>> def compute_feature(X):
    >>>     F = np.mean(X, axis = 1)
    >>>     return F
    X : array-like shape [n_segments, segment_width, n_variables]
    F : array-like [n_segments, n_features]
    The number of features returned (n_features) must be >= 1

    .. note:: ``np.atleast_3d`` is used if accessing the third dimension, as some datasets will
    have only a single time series variable. See ``hist4`` as an example.

See hist for an example of a callable object

Examples
--------
>>> from seglearn.feature_functions import all_features
>>> from seglearn.transform import FeatureRep
>>> FeatureTransform = FeatureRep(features=all_features())

"""

# Author: David Burns
# License: BSD

import numpy as np
from scipy import stats


def base_features():
    """ Returns dictionary of some basic features that can be calculated for segmented time
    series data """
    features = {'mean': mean,
                'median': median,
                'abs_energy': abs_energy,
                'std': std,
                'var': var,
                'min': minimum,
                'max': maximum,
                'skew': skew,
                'kurt': kurt,
                'mse': mse,
                'mnx': mean_crossings}
    return features


def all_features():
    """ Returns dictionary of all features in the module

    .. note:: Some of the features (hist4, corr) are relatively expensive to compute
    """
    features = {'mean': mean,
                'median': median,
                'gmean': gmean,
                'hmean': hmean,
                'vec_sum': vec_sum,
                'abs_sum': abs_sum,
                'abs_energy': abs_energy,
                'std': std,
                'var': var,
                'mad': median_absolute_deviation,
                'variation': variation,
                'min': minimum,
                'max': maximum,
                'skew': skew,
                'kurt': kurt,
                'mean_diff': mean_diff,
                'mean_abs_diff': means_abs_diff,
                'mse': mse,
                'mnx': mean_crossings,
                'hist4': hist(),
                'corr': corr2,
                'mean_abs_value': mean_abs,
                'zero_crossings': zero_crossing(),
                'slope_sign_changes': slope_sign_changes(),
                'waveform_length': waveform_length,
                'emg_var': emg_var,
                'root_mean_square': root_mean_square,
                'willison_amplitude': willison_amplitude()}
    return features


def hudgins_features(threshold=0):
    """Return a dict of Hudgin's time domain features used for EMG time series classification."""
    return {
        'mean_abs_value': mean_abs,
        'zero_crossings': zero_crossing(threshold),
        'slope_sign_changes': slope_sign_changes(threshold),
        'waveform_length': waveform_length,
    }


def emg_features(threshold=0):
    """Return a dictionary of popular features used for EMG time series classification."""
    return {
        'mean_abs_value': mean_abs,
        'zero_crossings': zero_crossing(threshold),
        'slope_sign_changes': slope_sign_changes(threshold),
        'waveform_length': waveform_length,
        'integrated_emg': abs_sum,
        'emg_var': emg_var,
        'simple square integral': abs_energy,
        'root_mean_square': root_mean_square,
        'willison_amplitude': willison_amplitude(threshold),
    }


def mean(X):
    """ statistical mean for each variable in a segmented time series """
    return np.mean(X, axis=1)


def median(X):
    """ statistical median for each variable in a segmented time series """
    return np.median(X, axis=1)


def gmean(X):
    """ geometric mean for each variable """
    return stats.gmean(X, axis=1)


def hmean(X):
    """ harmonic mean for each variable """
    return stats.hmean(X, axis=1)


def vec_sum(X):
    """ vector sum of each variable """
    return np.sum(X, axis=1)


def abs_sum(X):
    """ sum of absolute values """
    return np.sum(np.abs(X), axis=1)


def abs_energy(X):
    """ absolute sum of squares for each variable """
    return np.sum(X * X, axis=1)


def std(X):
    """ statistical standard deviation for each variable in a segmented time series """
    return np.std(X, axis=1)


def var(X):
    """ statistical variance for each variable in a segmented time series """
    return np.var(X, axis=1)


def median_absolute_deviation(X):
    """ median absolute deviation for each variable in a segmented time series """
    if hasattr(stats, 'median_abs_deviation'):
        return stats.median_abs_deviation(X, axis=1)
    else:
        return stats.median_absolute_deviation(X, axis=1)


def variation(X):
    """ coefficient of variation """
    return stats.variation(X, axis=1)


def minimum(X):
    """ minimum value for each variable in a segmented time series """
    return np.min(X, axis=1)


def maximum(X):
    """ maximum value for each variable in a segmented time series """
    return np.max(X, axis=1)


def skew(X):
    """ skewness for each variable in a segmented time series """
    return stats.skew(X, axis=1)


def kurt(X):
    """ kurtosis for each variable in a segmented time series """
    return stats.kurtosis(X, axis=1)


def mean_diff(X):
    """ mean temporal derivative """
    return np.mean(np.diff(X, axis=1), axis=1)


def means_abs_diff(X):
    """ mean absolute temporal derivative """
    return np.mean(np.abs(np.diff(X, axis=1)), axis=1)


def mse(X):
    """ computes mean spectral energy for each variable in a segmented time series """
    return np.mean(np.square(np.abs(np.fft.fft(X, axis=1))), axis=1)


def mean_crossings(X):
    """ Computes number of mean crossings for each variable in a segmented time series """
    X = np.atleast_3d(X)
    N = X.shape[0]
    D = X.shape[2]
    mnx = np.zeros((N, D))
    for i in range(D):
        pos = X[:, :, i] > 0
        npos = ~pos
        c = (pos[:, :-1] & npos[:, 1:]) | (npos[:, :-1] & pos[:, 1:])
        mnx[:, i] = np.count_nonzero(c, axis=1)
    return mnx


class hist(object):
    """ histogram for each variable in a segmented time series

    .. note:: this feature is expensive to compute with the current implementation
    """

    def __init__(self, bins=4):
        if bins < 2:
            raise ValueError("hist requires bins >= 2")
        self.bins = bins

    def __call__(self, X):
        X = np.atleast_3d(X)
        N = X.shape[0]
        D = X.shape[2]
        histogram = np.zeros((N, D * self.bins))
        for i in np.arange(N):
            for j in np.arange(D):
                # for each variable, advance by bins
                histogram[i, (j * self.bins):((j + 1) * self.bins)] = \
                    np.histogram(X[i, :, j], bins=self.bins, density=True)[0]

        return histogram

    def __repr__(self):
        return "%s(bins=%s)" % (self.__class__.__name__, self.bins)


def corr2(X):
    """ computes correlations between all variable pairs in a segmented time series

    .. note:: this feature is expensive to compute with the current implementation, and cannot be
    used with univariate time series
    """
    X = np.atleast_3d(X)
    N = X.shape[0]
    D = X.shape[2]

    if D == 1:
        return np.zeros(N, dtype=float)

    trii = np.triu_indices(D, k=1)
    DD = len(trii[0])
    r = np.zeros((N, DD))
    for i in np.arange(N):
        rmat = np.corrcoef(X[i])  # get the ith window from each signal, result will be DxD
        r[i] = rmat[trii]
    return r


def mean_abs(X):
    """ statistical mean of the absolute values for each variable in a segmented time series """
    return np.mean(np.abs(X), axis=1)


class zero_crossing(object):
    """ number of zero crossings among two consecutive samples above a certain threshold for each
    variable in the segmented time series"""

    def __init__(self, threshold=0):
        self.threshold = threshold

    def __call__(self, X):
        sign = np.heaviside(-1 * X[:, :-1] * X[:, 1:], 0)
        abs_diff = np.abs(np.diff(X, axis=1))
        return np.sum(sign * abs_diff >= self.threshold, axis=1, dtype=X.dtype)

    def __repr__(self):
        return "%s(threshold=%s)" % (self.__class__.__name__, self.threshold)


class slope_sign_changes(object):
    """ number of changes between positive and negative slope among three consecutive samples
    above a certain threshold for each variable in the segmented time series"""

    def __init__(self, threshold=0):
        self.threshold = threshold

    def __call__(self, X):
        change = (X[:, 1:-1] - X[:, :-2]) * (X[:, 1:-1] - X[:, 2:])
        return np.sum(change >= self.threshold, axis=1, dtype=X.dtype)

    def __repr__(self):
        return "%s(threshold=%s)" % (self.__class__.__name__, self.threshold)


def waveform_length(X):
    """ cumulative length of the waveform over a segment for each variable in the segmented time
    series """
    return np.sum(np.abs(np.diff(X, axis=1)), axis=1)


def root_mean_square(X):
    """ root mean square for each variable in the segmented time series """
    segment_width = X.shape[1]
    return np.sqrt(np.sum(X * X, axis=1) / segment_width)


def emg_var(X):
    """ variance (assuming a mean of zero) for each variable in the segmented time series
    (equals abs_energy divided by (seg_size - 1)) """
    segment_width = X.shape[1]
    return np.sum(X * X, axis=1) / (segment_width - 1)


class willison_amplitude(object):
    """ the Willison amplitude for each variable in the segmented time series """

    def __init__(self, threshold=0):
        self.threshold = threshold

    def __call__(self, X):
        segment_size = X.shape[1]
        return np.sum(np.abs(np.diff(X, axis=1)) >= self.threshold, axis=1)

    def __repr__(self):
        return "%s(threshold=%s)" % (self.__class__.__name__, self.threshold)
