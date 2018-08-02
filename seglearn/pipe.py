'''
This module is an sklearn compatible pipeline for machine learning
time series data and sequences using a sliding window segmentation
'''
# Author: David Burns
# License: BSD

from .transform import XyTransformerMixin


from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.externals import six

class Pype(Pipeline):
    #todo: handle steps with None

    def __init__(self, steps, scorer = None, memory = None):
        super(Pype, self).__init__(steps, memory)
        self.scorer = scorer

    def fit(self, X, y=None, **fit_params):
        """
        Fit the model

        Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.
        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : Pipeline
            This estimator
        """
        Xt, yt, fit_params = self._fit(X, y, **fit_params)

        self.N_train = len(yt)

        if self._final_estimator is not None:
            fitres = self._final_estimator.fit(Xt, yt, **fit_params)
            if hasattr(fitres, 'history'):
                self.history = fitres

        return self


    def _fit(self, X, y = None, **fit_params):
        self.steps = list(self.steps)
        self._validate_steps()

        fit_params_steps = dict((name, {}) for name, step in self.steps
                                if step is not None)
        for pname, pval in six.iteritems(fit_params):
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval

        Xt = X
        yt = y

        for step_idx, (name, transformer) in enumerate(self.steps[:-1]): #iterate through all but last
            if transformer is None:
                pass
            else:
                # not doing cloning for now...
                if isinstance(transformer, XyTransformerMixin):
                    Xt, yt, _ = transformer.fit_transform(Xt, yt, sample_weight=None, **fit_params_steps[name])
                else:
                    Xt = transformer.fit_transform(Xt, yt, **fit_params_steps[name])

        if self._final_estimator is None:
                return Xt, yt, {}
        return Xt, yt, fit_params_steps[self.steps[-1][0]]

    def _transform(self, X, y = None, sample_weight = None):
        Xt = X
        yt = y
        swt = sample_weight

        for name, transformer in self.steps[:-1]:  # iterate through all but last
            if isinstance(transformer, XyTransformerMixin):
                Xt, yt, swt = transformer.transform(Xt, yt, swt)
            else:
                Xt = transformer.transform(Xt)

        return Xt, yt, swt



    def transform(self, X, y = None):
        """
        Apply transforms, and transform with the final estimator
        This also works where final estimator is ``None``: all prior
        transformations are applied.

        Parameters
        ----------
        X : iterable
            Data to transform. Must fulfill input requirements of first step
            of the pipeline.
        y : array-like
            Target

        Returns
        -------
        Xt : array-like, shape = [n_samples, n_transformed_features]
            Transformed data
        yt : array-like, shape = [n_samples]
            Transformed target
        """
        Xt, yt, _ = self._transform(X, y)

        if isinstance(self._final_estimator, XyTransformerMixin):
            Xt, yt, _ = self._final_estimator.transform(Xt, yt)
        else:
            Xt = self._final_estimator.transform(Xt)

        return Xt, yt


    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit the model and transform with the final estimator
        Fits all the transforms one after the other and transforms the
        data, then uses fit_transform on transformed data with the final
        estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.
        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        Xt : array-like, shape = [n_samples, n_transformed_features]
            Transformed samples
        yt : array-like, shape = [n_samples]
            Transformed target
        """

        Xt, yt, fit_params = self._fit(X, y, **fit_params)

        if isinstance(self._final_estimator, XyTransformerMixin):
            Xt, yt, _ = self._final_estimator.fit_transform(Xt, yt)
        else:
            if hasattr(self._final_estimator, 'fit_transform'):
                Xt = self._final_estimator.fit_transform(Xt, yt)
            else:
                self._final_estimator.fit(Xt, yt)
                Xt = self._final_estimator.transform(Xt)

        self.N_fit = len(yt)

        return Xt, yt


    def predict(self, X):
        """
        Apply transforms to the data, and predict with the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        yp : array-like
            Predicted transformed target
        """
        Xt, _ , _ = self._transform(X)
        return self._final_estimator.predict(Xt)

    def transform_predict(self, X, y):
        """
        Apply transforms to the data, and predict with the final estimator.
        Unlike predict, this also returns the transformed target

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.
        y : array-like
            target

        Returns
        -------
        yt : array-like
            Transformed target
        yp : array-like
            Predicted transformed target
        """
        Xt, yt, _ = self._transform(X, y)
        yp = self._final_estimator.predict(Xt)
        return yt, yp


    def score(self, X, y=None, sample_weight=None):
        """
        Apply transforms, and score with the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.
        y : iterable, default=None
            Targets used for scoring. Must fulfill label requirements for all
            steps of the pipeline.
        sample_weight : array-like, default=None
            If not None, this argument is passed as ``sample_weight`` keyword
            argument to the ``score`` method of the final estimator.

        Returns
        -------
        score : float
        """

        Xt, yt, swt = self._transform(X, y, sample_weight)

        self.N_test = len(yt)

        score_params = {}
        if swt is not None:
            score_params['sample_weight'] = swt

        if self.scorer is None:
            return self._final_estimator.score(Xt, yt, **score_params)
        else:
            return self.scorer(self._final_estimator, Xt, yt, **score_params)


    def predict_proba(self, X):
        """
        Apply transforms, and predict_proba of the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_proba : array-like, shape = [n_samples, n_classes]
            Predicted probability of each class
        """
        Xt, _ , _ = self._transform(X)
        return self._final_estimator.predict_proba(Xt)

    def decision_function(self, X):
        """
        Apply transforms, and decision_function of the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_score : array-like, shape = [n_samples, n_classes]
        """
        Xt, _, _ = self._transform(X)
        return self._final_estimator.decision_function(Xt)

    def predict_log_proba(self, X):
        """
        Apply transforms, and predict_log_proba of the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_score : array-like, shape = [n_samples, n_classes]
        """
        Xt, _ , _ = self._transform(X)
        return self._final_estimator.predict_log_proba(Xt)

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self
        """
        items = self.steps
        names, _ = zip(*items)

        keys = list(six.iterkeys(params))

        for name in keys:
            if '__' not in name and name in names:
                # replace an estimator
                self._replace_estimator('steps', name, params.pop(name))

            if callable(params[name]):
                # use a callable or function to set parameters
                params[name] = params[name](params)

            elif params[name] in keys:
                # set one arg from another
                params[name] = params[params[name]]

        BaseEstimator.set_params(self, **params)
        return self


# class SegPipe(BaseEstimator):
#     '''
#     The pipeline supports learning multi-variate time series with or without relational contextual variables (see introduction).
#
#     The purpose of this pipeline is to assemble the segmentation, feature extraction (optional), feature processing (optional), and final estimator steps into a single pipeline that can be cross validated together setting different parameters.
#
#     The pipeline is applied in 2 steps, when methods ``fit``, ``transform``, ``predict``, or ``score`` is called
#
#     Step 1: sliding window segmentation performed by the ``segmenter`` object.
#
#     Step 2: ``est`` estimator pipeline (or estimator) called
#
#     The time series data (X) is input to the pipeline as a numpy object array, length n_series. The first column holds the time series data, and second column (optional) can hold relational contextual variables. Each element of the time series data is array-like with shape [n_samples, n_time_variables]. n_samples can be different for each element (series). Each element of the contextual variable data is array-like with shape [n_context_variables]. The ``util`` module has functions to help create the necessary data structure.
#
#     The target y can be like a contextual variable, with one value per time series, in which case ``SegmentX`` should be used as the segmenter. Or if the target variable y is itself a time series, than ``SegmentXY`` should be used.
#
#     The API for setting parameters for the segmenter and ``est`` pipelines are similar to sklearn but slightly different. It is compatible with sklearn parameter optimization tools (eg GridSearchCV). See the set_params method, and examples for details.
#
#     Parameters
#     ----------
#     est : sklearn estimator or pipeline or None
#         | for feature processing (optional) and applying the final estimator
#         | if None, only segmentation and expansion is done
#     segmenter : object
#         Segmentation transformer to convert time series data to segmented time series data
#     shuffle : bool, optional
#         shuffle the segments before fitting the ``est`` pipeline (recommended)
#     scorer : callable, optional
#         scorer callable made with sklearn.metrics.make_scorer, can only return 1 score
#     random_state : int, default = None
#         Randomized segment shuffling will return different results for each call to ``fit``. If you have set ``shuffle`` to True and want the same result with each call to ``fit``, set ``random_state`` to an integer.
#
#
#     Attributes
#     ----------
#     N_train : number of training segments - available after calling fit method
#     N_test : number of testing segments - available after calling predict, or score methods
#
#     Examples
#     --------
#
#     >>> from seglearn.transform import FeatureRep
#     >>> from seglearn.feature_functions import mean, var, std, skew
#     >>> from seglearn.pipe import SegPipe
#     >>> from seglearn.datasets import load_watch
#     >>> from sklearn.pipeline import Pipeline
#     >>> from sklearn.ensemble import RandomForestClassifier
#     >>> data = load_watch()
#     >>> X = data['X']
#     >>> y = data['y']
#     >>> fts = {'mean': mean, 'var': var, 'std': std, 'skew': skew}
#     >>> est = Pipeline([('ftr', FeatureRep(features = fts)),('rf',RandomForestClassifier())])
#     >>> pipe = SegPipe(est)
#     >>> pipe.fit(X, y)
#     >>> print(pipe.score(X, y))
#
#     '''
#     def __init__(self, est, segmenter = SegmentX(), scorer = None, shuffle = True, random_state = None):
#         self.est = est
#         self.segmenter = segmenter
#         self.scorer = scorer
#         self.shuffle = shuffle
#         self.random_state = random_state
#
#     def fit(self, X, y, **fit_params):
#         '''
#         Fit the data.
#         Applies fit to ``feed`` pipeline, segment expansion, and fit on ``est`` pipeline.
#
#         Parameters
#         ----------
#         X : array-like, shape [n_series, ...]
#            Time series data and (optionally) contextual data created as per ``make_ts_data``
#         y : array-like shape [n_series]
#             target vector
#         fit_params : optional parameters
#             passed to ``est`` pipe fit method
#
#         Returns
#         -------
#         self : object
#             Return self
#
#         '''
#         self._reset()
#         check_ts_data(X, y)
#         self.segmenter.fit(X, y, **fit_params)
#         X, y, _ = self.segmenter.transform(X, y)
#
#         self.N_train = len(y)
#
#         if self.shuffle is True:
#             X, y = self._shuffle(X, y)
#
#         fitres = self.est.fit(X, y, **fit_params)
#
#         # for keras scikit learn api
#         if hasattr(fitres,'history'):
#             self.history = fitres
#
#         return self
#
#     def transform(self, X, y = None):
#         '''
#         Applies transform to ``est`` pipeline.
#         If the target vector y is supplied, the segmentation expansion will be performed on it and returned.
#
#         Parameters
#         ----------
#         X : array-like, shape [n_series, ...]
#            Time series data and (optionally) contextual data created as per ``make_ts_data``
#         y : array-like shape [n_series], optional
#             target vector
#
#         Returns
#         -------
#         X : array-like, shape [n_segments, ]
#             transformed feature data
#         y : array-like, shape [n_segments]
#             expanded target vector
#         '''
#         check_is_fitted(self, 'N_train')
#         check_ts_data(X, y)
#         X, y, _ = self.segmenter.transform(X, y)
#         X = self.est.transform(X)
#         return X, y
#
#
#     def fit_transform(self, X, y, **fit_params):
#         '''
#         Applies fit and transform methods to the full pipeline sequentially
#
#         Parameters
#         ----------
#         X : array-like, shape [n_series, ...]
#            Time series data and (optionally) contextual data created as per ``make_ts_data``
#         y : array-like shape [n_series], optional
#             target vector
#
#         Returns
#         -------
#         X : array-like, shape [n_segments, ]
#             transformed feature data
#         y : array-like, shape [n_segments]
#             expanded target vector
#         '''
#         self.fit(X,y, **fit_params)
#         return self.transform(X,y)
#
#
#
#     def predict(self, X, y):
#         '''
#         Applies transform to ``feed`` pipeline, segment expansion, and predict on ``est`` pipeline.
#
#         Parameters
#         ----------
#         X : array-like, shape [n_series, ...]
#            Time series data and (optionally) contextual data created as per ``make_ts_data``
#         y : array-like shape [n_series], optional
#             target vector
#
#         Returns
#         -------
#         y : array like shape [n_segments]
#             expanded target vector
#         yp : array like shape [n_segments]
#             predicted (expanded) target vector
#
#         '''
#         check_is_fitted(self, 'N_train')
#         check_ts_data(X, y)
#         X, y, _ = self.segmenter.transform(X, y)
#         yp = self.est.predict(X)
#         self.N_test = len(y)
#         return y, yp
#
#
#     def score(self, X, y, sample_weight = None):
#         '''
#         Applies transform and score with the final estimator or scorer function if set as init parameter
#
#         Parameters
#         ----------
#         X : array-like, shape [n_series, ...]
#            Time series data and (optionally) contextual data created as per ``make_ts_data``
#         y : array-like shape [n_series], optional
#             target vector
#         sample_weight : array-like shape [n_series], default = None
#             If not none, this is expanded from a by-series to a by-segment representation and passed as a ``sample_weight`` keyword argument to the ``score`` method of the ``est`` pipeline.
#
#         Returns
#         -------
#         score : float
#         '''
#         check_is_fitted(self, 'N_train')
#         check_ts_data(X, y)
#         X, y, sample_weight = self.segmenter.transform(X, y, sample_weight)
#
#         score_params = {}
#         if sample_weight is not None:
#             score_params['sample_weight'] = sample_weight
#
#         self.N_test = len(y)
#
#         if self.scorer is None:
#             return self.est.score(X, y, **score_params)
#         else:
#             return self.scorer(self.est, X, y, **score_params)
#
#
#     def set_params(self, **params):
#         '''
#         Set the parameters of of the pipeline.
#
#         Parameters
#         ----------
#         params : dict
#             | parameter keys for the ``feed`` pipeline are preceeded by ``f$``
#             | parameter keys for the ``est`` pipeline are preceded by ``e$``
#             | parameter keys can be set to the value of another parameter using the key for that parameter as a value
#
#         Returns
#         -------
#         self : object
#             Returns self
#
#         '''
#         keys = list(params.keys())
#         for key in params:
#             if params[key] in keys:
#                 params[key] = params[params[key]]
#
#         seg_params = {key[2:]:params[key] for key in params if 's$' in key}
#         est_params = {key:params[key] for key in params if 's$' not in key}
#         self.segmenter.set_params(**seg_params)
#         self.est.set_params(**est_params)
#         return self
#
#     # def get_params(self, deep=True):
#     # not yet implemented
#     #     feed_params = self.feed.get_params()
#     #     feed_params = {'f$'+key:feed_params[key] for key in feed_params}
#     #
#     #     est_params = self.est.get_params()
#     #     est_params = {'e$' + key: est_params[key] for key in est_params}
#     #
#     #     return dict(list(feed_params.items()) + list(est_params.items()))
#
#
#     def _reset(self):
#         ''' Resets internal data-dependent state of the transformer. __init__ parameters not touched. '''
#         if hasattr(self,'N_train'):
#             del self.N_train
#         if hasattr(self, 'N_test'):
#             del self.N_test
#
#     def _shuffle(self, X, y):
#         ''' shuffles X and y '''
#         if len(y) > 1:
#             check_random_state(self.random_state)
#             ind = np.arange(len(y), dtype=np.int)
#             np.random.shuffle(ind)
#             X = X[ind]
#             y = y[ind]
#         return X, y
#
#
#
#
#
