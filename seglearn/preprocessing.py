"""
This module is for preprocessing time series data.
"""
# Author: David Burns
# License: BSD

import numpy as np
from sklearn.base import BaseEstimator

from .base import TS_Data
from .util import get_ts_data_parts, check_ts_data_with_ts_target
from .transform import XyTransformerMixin, expand_variables_to_segments

__all__ = ['TargetRunLengthEncoder']

class TargetRunLengthEncoder(BaseEstimator, XyTransformerMixin):
    """
    Takes a data set with a categorical target variable encoded as a time series and transforms it
    with run length encoding (RLE) of the target variable

    RLE finds contiguous runs of the same target value within the input data and derives the
    transformed data set from the amalgum of all contiguous runs of all target classes from all
    series in the input data.

    This is useful for generating "pure" series with no mixing of target variables from datasets
    that encode the target variable as a series (e.g. MHEALTH and PAMAP2)

    Note that ``seglearn`` can handle datasets with target variables encoded as a series natively
    (using ``SegmentXY``) and so this preprocessing is not required but may be helpful for some tasks.
    Effectively it will let you use ``SegmentX`` on datasets that would otherwise require ``SegmentXY``.

    Parameters
    ----------
    min_length : integer > 1
        minimum number of samples in a run for it to be included in the transformed data
    """

    def __init__(self, min_length = 200):
        self.min_length = min_length
        self._validate_params()

    def _validate_params(self):
        if not self.min_length > 1:
            raise ValueError("min_length must be >1 (was %d)" % self.min_length)

    def fit(self, X, y=None):
        """
        Fit the transform

        Parameters
        ----------
        X : array-like, shape [n_series, ...]
            Time series data and (optionally) contextual data
        y : None
            There is no need of a target in a transformer, yet the pipeline API requires
            this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        check_ts_data_with_ts_target(X, y)
        return self

    def transform(self, X, y, sample_weight=None):
        """
        Transforms the time series data with run length encoding of the target variable
        Note this transformation changes the number of samples in the data
        If sample_weight is provided, it is transformed to align to the new target encoding


        Parameters
        ----------
        X : array-like, shape [n_series, ...]
           Time series data and (optionally) contextual data
        y : array-like shape [n_series, ...]
            target variable encoded as a time series
        sample_weight : array-like shape [n_series], default = None
            sample weights

        Returns
        -------
        Xt : array-like, shape [n_rle_series, ]
            transformed time series data
        yt : array-like, shape [n_rle_series]
            target values for each series
        sample_weight_new : array-like shape [n_rle_series]
            sample weights
        """
        check_ts_data_with_ts_target(X, y)

        Xt, Xc = get_ts_data_parts(X)
        N = len(Xt)  # number of time series

        # transformed data
        yt = []
        Xtt = []
        swt = sample_weight
        Nt = []

        for i in range(N):
            Xi, yi = self._transform(Xt[i], y[i])
            yt+=yi
            Xtt+=Xi
            Nt.append(len(yi)) # number of contiguous class instances

        if Xc is not None:
            Xct = expand_variables_to_segments(Xc, Nt)
            Xtt = TS_Data(Xtt, Xct)

        if sample_weight is not None:
            swt = expand_variables_to_segments(sample_weight, Nt)

        return Xtt, yt, swt

    def _rle(self, a):
        """
        rle implementation credit to Thomas Browne from his SOF post Sept 2015

        Parameters
        ----------
        a : array, shape[n,]
            input vector

        Returns
        -------
        z : array, shape[nt,]
            run lengths
        p : array, shape[nt,]
            start positions of each run
        ar : array, shape[nt,]
            values for each run
        """
        ia = np.asarray(a)
        n = len(ia)
        y = np.array(ia[1:] != ia[:-1])  # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)  # must include last element posi
        z = np.diff(np.append(-1, i))  # run lengths
        p = np.cumsum(np.append(0, z))[:-1]  # positions
        return z, p, ia[i]

    def _transform(self, X, y):
        """
        Transforms single series
        """
        z, p, y_rle = self._rle(y)
        p = np.append(p, len(y))
        big_enough = p[1:] - p[:-1] >= self.min_length
        Xt = []

        for i in range(len(y_rle)):
            if big_enough[i]:
                Xt.append(X[p[i]:p[i+1]])

        yt = y_rle[big_enough].tolist()
        return Xt, yt







