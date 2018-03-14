'''
This module is an sklearn compatible pipeline for machine learning
time series data and sequences using a sliding window segmentation
'''
# Author: David Burns
# License: BSD

from .transform import SegmentX
from .util import check_ts_data

import numpy as np
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils.validation import check_is_fitted

class SegPipe(_BaseComposition):
    '''
    The pipeline supports learning multi-variate time series with or without relational contextual variables (see introduction).

    The purpose of this pipeline is to assemble the segmentation, feature extraction (optional), feature processing (optional), and final estimator steps into a single pipeline that can be cross validated together setting different parameters.

    The pipeline is applied in 2 steps, when methods ``fit``, ``transform``, ``predict``, or ``score`` is called

    Step 1: sliding window segmentation performed by the ``segmenter`` object.

    Step 2: ``est`` estimator pipeline (or estimator) called

    The time series data (X) is input to the pipeline as a numpy object array, length n_series. The first column holds the time series data, and second column (optional) can hold relational contextual variables. Each element of the time series data is array-like with shape [n_samples, n_time_variables]. n_samples can be different for each element (series). Each element of the contextual variable data is array-like with shape [n_context_variables]. The ``util`` module has functions to help create the necessary data structure.

    The target y can be like a contextual variable, with one value per time series, in which case ``SegmentX`` should be used as the segmenter. Or if the target variable y is itself a time series, than ``SegmentXY`` should be used.

    The API for setting parameters for the segmenter and ``est`` pipelines are similar to sklearn but slightly different. It is compatible with sklearn parameter optimization tools (eg GridSearchCV). See the set_params method, and examples for details.

    Parameters
    ----------
    est : sklearn estimator or pipeline or None
        | for feature processing (optional) and applying the final estimator
        | if None, only segmentation and expansion is done
    segmenter : object
        Segmentation transformer to convert time series data to segmented time series data
    shuffle : bool, optional
        shuffle the segments before fitting the ``est`` pipeline (recommended)
    scorer : callable, optional
        scorer callable made with sklearn.metrics.make_scorer, can only return 1 score


    Attributes
    ----------
    N_train : number of training segments - available after calling fit method
    N_test : number of testing segments - available after calling predict, or score methods

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
    def __init__(self, est, segmenter = SegmentX(), shuffle = True, scorer = None):
        self.est = est
        self.segmenter = segmenter
        self.shuffle = shuffle
        self.scorer = scorer

    def fit(self, X, y, **fit_params):
        '''
        Fit the data.
        Applies fit to ``feed`` pipeline, segment expansion, and fit on ``est`` pipeline.

        Parameters
        ----------
        X : array-like, shape [n_series, ...]
           Time series data and (optionally) contextual data created as per ``make_ts_data``
        y : array-like shape [n_series]
            target vector
        fit_params : optional parameters
            passed to ``est`` pipe fit method

        Returns
        -------
        self : object
            Return self

        '''
        self._reset()
        check_ts_data(X, y)
        self.segmenter.fit(X, y, **fit_params)
        X, y, _ = self.segmenter.transform(X, y)

        self.N_train = len(y)

        if self.shuffle is True:
            X, y = self._shuffle(X, y)

        fitres = self.est.fit(X, y, **fit_params)

        # for keras scikit learn api
        if hasattr(fitres,'history'):
            self.history = fitres

        return self

    def transform(self, X, y = None):
        '''
        Applies transform to ``est`` pipeline.
        If the target vector y is supplied, the segmentation expansion will be performed on it and returned.

        Parameters
        ----------
        X : array-like, shape [n_series, ...]
           Time series data and (optionally) contextual data created as per ``make_ts_data``
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
        check_ts_data(X, y)
        X, y, _ = self.segmenter.transform(X, y)
        X = self.est.transform(X)
        return X, y


    def fit_transform(self, X, y, **fit_params):
        '''
        Applies fit and transform methods to the full pipeline sequentially

        Parameters
        ----------
        X : array-like, shape [n_series, ...]
           Time series data and (optionally) contextual data created as per ``make_ts_data``
        y : array-like shape [n_series], optional
            target vector

        Returns
        -------
        X : array-like, shape [n_segments, ]
            transformed feature data
        y : array-like, shape [n_segments]
            expanded target vector
        '''
        self.fit(X,y, **fit_params)
        return self.transform(X,y)



    def predict(self, X, y):
        '''
        Applies transform to ``feed`` pipeline, segment expansion, and predict on ``est`` pipeline.

        Parameters
        ----------
        X : array-like, shape [n_series, ...]
           Time series data and (optionally) contextual data created as per ``make_ts_data``
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
        check_ts_data(X, y)
        X, y, _ = self.segmenter.transform(X, y)
        yp = self.est.predict(X)
        self.N_test = len(y)
        return y, yp


    def score(self, X, y, sample_weight = None):
        '''
        Applies transform and score with the final estimator or scorer function if set as init parameter

        Parameters
        ----------
        X : array-like, shape [n_series, ...]
           Time series data and (optionally) contextual data created as per ``make_ts_data``
        y : array-like shape [n_series], optional
            target vector
        sample_weight : array-like shape [n_series], default = None
            If not none, this is expanded from a by-series to a by-segment representation and passed as a ``sample_weight`` keyword argument to the ``score`` method of the ``est`` pipeline.

        Returns
        -------
        score : float
        '''
        check_is_fitted(self, 'N_train')
        check_ts_data(X, y)
        X, y, sample_weight = self.segmenter.transform(X, y, sample_weight)

        score_params = {}
        if sample_weight is not None:
            score_params['sample_weight'] = sample_weight

        self.N_test = len(y)

        if self.scorer is None:
            return self.est.score(X, y, **score_params)
        else:
            return self.scorer(self.est, X, y, **score_params)


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

        seg_params = {key[2:]:params[key] for key in params if 's$' in key}
        est_params = {key:params[key] for key in params if 's$' not in key}
        self.segmenter.set_params(**seg_params)
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
        if hasattr(self, 'N_test'):
            del self.N_test

    def _shuffle(self, X, y):
        ''' shuffles X and y '''
        ind = np.arange(len(y), dtype=np.int)
        np.random.shuffle(ind)
        X = X[ind]
        y = y[ind]
        return X, y


