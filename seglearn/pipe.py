'''
This module is an sklearn compatible pipeline for machine learning
time series data and sequences using a sliding window segmentation
'''
# Author: David Burns
# License: BSD

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.externals import six

from .transform import XyTransformerMixin


class Pype(Pipeline):
    '''
    This pipeline extends the sklearn Pipeline to support transformers that change X, y,
    sample_weight, and the number of samples.

    It also adds some new options for setting hyper-parameters with callables and in reference to
    other parameters (see examples).

    Parameters
    ----------
    steps : list
        List of (name, transform) tuples (implementing fit/transform) that are chained, in the
        order in which they are chained, with the last object an estimator.
    scorer : sklearn scorer object
    memory : currently not implemented

    Attributes
    ----------
    N_train : number of training samples - available after calling fit method
    N_test : number of testing samples - available after calling predict, or score methods

    Examples
    --------

    >>> from seglearn.transform import FeatureRep, SegmentX
    >>> from seglearn.pipe import Pype
    >>> from seglearn.datasets import load_watch
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.preprocessing import StandardScaler
    >>> data = load_watch()
    >>> X = data['X']
    >>> y = data['y']
    >>> pipe = Pype([('segment', SegmentX()),
    >>>              ('features', FeatureRep()),
    >>>              ('scaler', StandardScaler()),
    >>>              ('rf', RandomForestClassifier())])
    >>> pipe.fit(X, y)
    >>> print(pipe.score(X, y))

    '''

    # todo: handle steps with None
    def __init__(self, steps, scorer=None, memory=None):
        super(Pype, self).__init__(steps, memory)
        self.scorer = scorer
        self.N_train = None
        self.N_test = None
        self.N_fit = None
        self.history = None

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

    def _fit(self, X, y=None, **fit_params):
        self.steps = list(self.steps)
        self._validate_steps()

        fit_params_steps = dict((name, {}) for name, step in self.steps
                                if step is not None)
        for pname, pval in six.iteritems(fit_params):
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval

        Xt = X
        yt = y

        # iterate through all but last
        for step_idx, (name, transformer) in enumerate(self.steps[:-1]):
            if transformer is None:
                pass
            else:
                # not doing cloning for now...
                if isinstance(transformer, XyTransformerMixin):
                    Xt, yt, _ = transformer.fit_transform(Xt, yt, sample_weight=None,
                                                          **fit_params_steps[name])
                else:
                    Xt = transformer.fit_transform(Xt, yt, **fit_params_steps[name])

        if self._final_estimator is None:
            return Xt, yt, {}
        return Xt, yt, fit_params_steps[self.steps[-1][0]]

    def _transform(self, X, y=None, sample_weight=None):
        Xt = X
        yt = y
        swt = sample_weight

        for name, transformer in self.steps[:-1]:  # iterate through all but last
            if isinstance(transformer, XyTransformerMixin):
                Xt, yt, swt = transformer.transform(Xt, yt, swt)
            else:
                Xt = transformer.transform(Xt)

        return Xt, yt, swt

    def transform(self, X, y=None):
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
        Xt, _, _ = self._transform(X)
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
        Xt, _, _ = self._transform(X)
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
        Xt, _, _ = self._transform(X)
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
