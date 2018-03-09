'''
This module is an sklearn compatible pipeline for machine learning
time series data and sequences using a sliding window segmentation
'''
# Author: David Burns
# License: BSD

from .util import make_ts_data, get_ts_data_parts

import numpy as np
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline

class SegPipe(_BaseComposition):
    '''
    The pipeline supports learning multi-variate time series with or without relational static variables (see introduction).

    The purpose of this pipeline is to assemble the segmentation, feature extraction (optional), feature processing (optional), and final estimator steps into a single pipeline that can be cross validated together setting different parameters.

    The pipeline is applied in three steps.

    The first step is applied by the ``feed`` pipeline (or transformer) that performs the sliding window segmentation, and optionally any feature extraction. Feature extraction is not required for estimators that learn the segments directly (eg a recurrent neural network). Feature extraction can be helpful for classifiers such as RandomForest or SVC that learn a feature representation of the segments.

    The second step expands the target vector (y), aligning it with the segments or the feature representation of the segments computed during the first step. The segmentation changes the effective number of samples available to the estimator, which is not supported within native sklearn pipelines and is the motivation for this extension.

    The third step applies the ``est`` estimator pipeline (or estimator) that performs additional feature processing (optional) and applies the final estimator for classification or regression.

    The time series data (X) is input to the pipeline as a numpy object array. The first column holds the time series data, and subsequent columns can hold relational static variables. Each element of the time series data is array-like with shape [n_samples, n_variables]. n_samples can be different for each element (series). The ``util`` module has functions to help create the necessary data structure.

    The API for setting parameters for the ``feed`` and ``est`` pipelines are different, but compatible with sklearn parameter optimization tools (eg GridSearchCV). See the set_params method, and examples for details.

    Parameters
    ----------
    feed : Segment transformer or sklearn pipeline chaining Segment with a feature extractor
    est : sklearn estimator or pipeline
        for feature processing (optional) and applying the final estimator
        can also be None
    shuffle : bool, optional
        shuffle the segments before fitting the ``est`` pipeline (recommended)


    Attributes
    ----------
    f_labels : list of string feature labels (in order) corresponding to the computed features - available after calling fit method
    N_train : number of training segments - available after calling fit method
    N_test : number of testing segments - available after calling predict, transform, or score methods

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

    Todo
    ----
    - shuffling

    '''
    def __init__(self, feed, est, shuffle = True):
        self.feed = feed
        self.est = est
        self.shuffle = shuffle

    def fit(self, X, y, **est_fit_params):
        '''
        Fit the data.
        Applies fit to ``feed`` pipeline, segment expansion, and fit on ``est`` pipeline.

        Parameters
        ----------
        X : array-like, shape [n_series, ...]
           Time series data and (optionally) static data created as per ``make_ts_data``
        y : array-like shape [n_series]
            target vector
        est_fit_params : optional parameters
            passed to ``est`` pipe fit method

        Returns
        -------
        self : object
            Return self

        '''
        self._reset()

        self.feed.fit(X, y)
        X = self.feed.transform(X)

        if type(self.feed) is Pipeline:
            self.f_labels = self.feed._final_estimator.f_labels
        else:
            self.f_labels = self.feed.f_labels

        # make sure the number of labels corresponds to the number of columns in X
        # put this in testing
        # if self.f_labels is not None:
        #     assert len(self.f_labels) == np.row_stack(X['ts'][0]).shape[1]

        X, y = self._ts_expand(X, y)

        self.N_train = len(y)

        if self.shuffle is True:
            # todo: X, y = self._shuffle(X, y)
            pass

        if self.est is not None:
            self.est.fit(X, y, **est_fit_params)

    def transform(self, X, y = None):
        '''
        Applies transform to ``feed`` pipeline, segment expansion, and transform on ``est`` pipeline.
        If the target vector y is supplied, the segmentation expansion will be performed on it and returned.

        Parameters
        ----------
        X : array-like, shape [n_series, ...]
           Time series data and (optionally) static data created as per ``make_ts_data``
        y : array-like shape [n_series], optional
            target vector

        Returns
        -------
        X : array-like, shape [n_segments, ]
            transformed feature data
        y : array-like, shape [n_segments]
            expanded target vector
        '''
        check_is_fitted(self, 'N_train')
        X = self.feed.transform(X)
        X, y = self._ts_expand(X, y)


        self.N_test = len(X[0]) # to get Xt
        if self.est is not None:
            X = self.est.transform(X)

        return X, y


    def fit_transform(self, X, y):
        '''
        Applies fit and transform methods to the full pipeline sequentially

        Parameters
        ----------
        X : array-like, shape [n_series, ...]
           Time series data and (optionally) static data created as per ``make_ts_data``
        y : array-like shape [n_series], optional
            target vector

        Returns
        -------
        X : array-like, shape [n_segments, ]
            transformed feature data
        y : array-like, shape [n_segments]
            expanded target vector
        '''
        self.fit(X,y)
        return self.transform(X, y)


    def predict(self, X, y):
        '''
        Applies transform to ``feed`` pipeline, segment expansion, and predict on ``est`` pipeline.

        Parameters
        ----------
        X : array-like, shape [n_series, ...]
           Time series data and (optionally) static data created as per ``make_ts_data``
        y : array-like shape [n_series], optional
            target vector

        Returns
        -------
        y : array like shape [n_segments]
            expanded target vector
        yp : array like shape [n_segments]
            predicted (expanded) target vector

        '''
        check_is_fitted(self, 'N_train')

        X = self.feed.transform(X)
        X, y = self._ts_expand(X, y)
        yp = self.est.predict(X)
        self.N_test = len(y)
        return y, yp


    def score(self, X, y, sample_weight = None):
        '''
        Applies transform and score with the final estimator

        Parameters
        ----------
        X : array-like, shape [n_series, ...]
           Time series data and (optionally) static data created as per ``make_ts_data``
        y : array-like shape [n_series], optional
            target vector
        sample_weight : array-like shape [n_series], default = None
            If not none, this is expanded from a by-series to a by-segment representation and passed as a ``sample_weight`` keyword argument to the ``score`` method of the ``est`` pipeline.

        Returns
        -------
        score : float
        '''
        check_is_fitted(self, 'N_train')

        score_params = {}
        if sample_weight is not None:
            Xt, Xs = get_ts_data_parts(X)
            Nt = self._number_of_segments_per_series(Xt)
            score_params['sample_weight'] = self._expand_target_to_segments(sample_weight, Nt)

        X = self.feed.transform(X)
        X, y = self._ts_expand(X, y)

        self.N_test = len(y)
        return self.est.score(X, y, **score_params)


    def set_params(self, **params):
        '''
        Set the parameters of of the pipeline.

        Parameters
        ----------
        params : dict
            | parameter keys for the ``feed`` pipeline are preceeded by ``f$``
            | parameter keys for the ``est`` pipeline are preceded by ``e$``
            | parameter keys can be set to the value of another parameter using the key for that parameter as a value

        Returns
        -------
        self : object
            Returns self

        '''
        keys = list(params.keys())
        for key in params:
            if params[key] in keys:
                params[key] = params[params[key]]

        feed_params = {key[2:]:params[key] for key in params if 'f$' in key}
        est_params = {key[2:]:params[key] for key in params if 'e$' in key}
        self.feed.set_params(**feed_params)
        self.est.set_params(**est_params)
        return self

    # def get_params(self, deep=True):
    # not yet implemented
    #     feed_params = self.feed.get_params()
    #     feed_params = {'f$'+key:feed_params[key] for key in feed_params}
    #
    #     est_params = self.est.get_params()
    #     est_params = {'e$' + key: est_params[key] for key in est_params}
    #
    #     return dict(list(feed_params.items()) + list(est_params.items()))


    def _reset(self):
        ''' Resets internal data-dependent state of the transformer. __init__ parameters not touched. '''
        if hasattr(self,'N_train'):
            del self.N_train
            del self.f_labels
        if hasattr(self, 'N_test'):
            del self.N_test

    def _expand_target_to_segments(self, y, Nt):
        ''' expands variable vector v, by repeating each instance as specified in Nt '''
        y_e = np.concatenate([np.full(Nt[i], y[i]) for i in np.arange(len(y))])
        return y_e

    def _expand_static_variables_to_segments(self, v, Nt):
        N_v = len(np.atleast_1d(v[0]))
        return np.concatenate([np.full((Nt[i], N_v), v[i]) for i in np.arange(len(v))])

    def _number_of_segments_per_series(self, Xt):
        ''' returns the number of segments in each series of X '''
        Nt = [len(Xt[i]) for i in np.arange(len(Xt))]
        return Nt

    def _ts_expand(self, X, y = None):
        # if its a record array we need to expand 'ts' and 'h'
        Xt, Xs = get_ts_data_parts(X)
        Nt = self._number_of_segments_per_series(Xt)
        Xt_new = np.concatenate(Xt)

        ye = []
        if y is not None:
            ye = self._expand_target_to_segments(y, Nt)

        if Xs is None:
            return Xt_new, ye
        else:
            Xs_new = self._expand_static_variables_to_segments(Xs, Nt)
            X_new = make_ts_data(Xt_new, Xs_new)
            return X_new, ye

    def _shuffle(self, X, y):
        ''' shuffles X and y '''
        ind = np.arange(len(y), dtype=np.int)
        np.random.shuffle(ind)
        X = X[ind]
        y = y[ind]
        return X, y


