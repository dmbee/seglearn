'''
This module is an sklearn compatible pipeline for machine learning
time series data and sequences using a sliding window segmentation
'''
# Author: David Burns
# License: BSD


import numpy as np
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline

class SegPipe(_BaseComposition):
    '''
    The pipeline supports learning multi-variate time series with or without relational static variables (see introduction).

    The purpose of this pipeline is to assemble the segmentation, feature extraction (optional), feature processing (optional), and final estimator steps into a single pipeline that can be cross validated together setting different parameters.

    The pipeline is applied in three steps.

    The first step is applied by the 'feed' pipeline (or transformer) that performs the sliding window segmentation, and optionally any feature extraction. Feature extraction is not required for estimators that learn the segments directly (eg a recurrent neural network). Feature extraction can be helpful for classifiers such as RandomForest or SVC that learn a feature representation of the segments.

    The second step expands the target vector (y), aligning it with the segments or segment features computed during the first step. The segmentation changes the effective number of samples available to the estimator, which is not supported within native sklearn pipelines and is the motivation for this extension.

    The third step applies the 'est' estimator pipeline (or estimator) that performs additional feature processing (optional) and applies the final estimator for classification or regression.

    The time series data (X) can be given to SegPipe methods in three formats: 1) a list of arrays 2) an object array, or 3) a recarray. The recarray format is required to support a combination time-series and static relational data. If using a recarray, the time series data must have name 'ts'. Each element of the time series data is array-like with shape [n_samples, segment_width, n_variables]. n_samples can be different for each element (series).

    The recarray data format is used by the 'feed' pipeline, and time series data input to SegPipe as an object array or list is converted to a recarray. At the second step (expansion), if the time-series data has been converted to a feature representation or has no static variables, it is converted back into a numpy object array for convenience.

    The API for setting parameters for the 'feed' and 'est' pipelines are different, but compatible with sklearn parameter optimization tools (eg GridSearchCV). See the set_params method, and examples for details.

    Parameters
    ----------
    feed : Segment transformer or sklearn pipeline chaining Segment with a feature extractor
    est : sklearn estimator or pipeline for feature processing (optional) and applying the final estimator
    shuffle : bool, optional
        shuffle the segments before fitting the 'est' pipeline (recommended)


    Attributes
    ----------
    f_labels : list of string feature labels (in order) corresponding to the computed features - available after calling fit method
    N_train : number of training segments - available after calling fit method
    N_test : number of testing segments - available after calling predict, transform, or score methods

    Examples
    --------

    >>> from seglearn.features import SegFeatures, mean, var, std, skew
    >>> from seglearn.pipe import SegPipe
    >>> from seglearn.segment import Segment
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
    - scoring
    - shuffling

    '''
    def __init__(self, feed, est, shuffle = True):
        self.feed = feed
        self.est = est
        self.shuffle = shuffle

    def fit(self, X, y, **est_fit_params):
        '''
        Fit the data.
        Applies fit to 'feed' pipeline, segment expansion, and fit on 'est' pipeline.

        Parameters
        ----------
        X : list, object array, or recarray shape [n_series, ...]
            Time series data and (optionally) static data
        y : array-like shape [n_series]
            target vector
        est_fit_params : optional parameters
            passed to 'est' pipe fit method

        Returns
        -------
        self : object
            Return self

        '''
        self._reset()
        X = self._check_data(X)

        self.feed.fit(X, y)
        X = self.feed.transform(X)

        if type(self.feed) is Pipeline:
            self.f_labels = self.feed._final_estimator.f_labels
        else:
            self.f_labels = self.feed.f_labels
        # make sure the number of labels corresponds to the number of columns in X
        if self.f_labels is not None:
            assert len(self.f_labels) == np.row_stack(X['ts'][0]).shape[1]

        X, y = self._ts_expand(X, y)
        self.N_train = len(X)

        if self.shuffle is True:
            # todo: X, y = self._shuffle(X, y)
            pass
        self.est.fit(X, y, **est_fit_params)

    def transform(self, X, y = None):
        '''
        Applies transform to 'feed' pipeline, segment expansion, and transform on 'est' pipeline.
        If the target vector y is supplied, the segmentation expansion will be performed on it and returned.

        Parameters
        ----------
        X : list, object array, or recarray shape [n_series, ...]
            Time series data and (optionally) static data
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
        X = self._check_data(X)
        X = self.feed.transform(X)
        X, y = self._ts_expand(X, y)
        self.N_test = len(X)
        X = self.est.transform(X)
        return X, y


    def predict(self, X, y):
        '''
        Applies transform to 'feed' pipeline, segment expansion, and predict on 'est' pipeline.

        Parameters
        ----------
        X : list, object array, or recarray shape [n_series, ...]
            Time series data and (optionally) static data
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

        X = self._check_data(X)
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
        X : list, object array, or recarray shape [n_series, ...]
            Time series data and (optionally) static data
        y : array-like shape [n_series], optional
            target vector
        sample_weight : array-like shape [n_series], default = None
            If not none, this is expanded from a by-series to a by-segment representation and passed as a ``sample_weight`` keyword argument to the ``score`` method of the ``est`` pipeline.

        Returns
        -------
        score : float
        '''
        check_is_fitted(self, 'N_train')
        X = self._check_data(X)

        score_params = {}
        if sample_weight is not None:
            Nt = self._number_of_segments_per_series(X['ts'])
            score_params['sample_weight'] = self._expand_series_variable_to_segments(sample_weight, Nt)

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

    def _check_data(self, X, y = None): #todo: check y
        ''' checks input data, returns recarray if list or object array given '''
        try:
            dnames = X.dtype.names
            assert 'ts' in dnames
            return X
        except:
            return np.core.records.fromarrays([X], names = ['ts'])

    def _expand_series_variable_to_segments(self, v, Nt):
        ''' expands variable vector v, by repeating each instance as specified in Nt '''
        v_e = np.concatenate([np.full(Nt[i], v[i]) for i in np.arange(len(v))])
        return v_e

    def _number_of_segments_per_series(self, X):
        ''' returns the number of segments in each series of X '''
        Nt = [len(X[i]) for i in np.arange(len(X))]
        return Nt

    def _ts_expand(self, X, y = None):
        # if its a record array we need to expand 'ts' and 'h'
        Nt = self._number_of_segments_per_series(X['ts'])
        ye = []
        if y is not None:
            ye = self._expand_series_variable_to_segments(y, Nt)

        Xt = np.concatenate(X['ts'])
        s_names = [h for h in X.dtype.names if h != 'ts']

        if len(s_names) == 0:
            return Xt, ye
        else:
            s_arrays = []
            for h in s_names:
                s_arrays.append(self._expand_series_variable_to_segments(X[h], Nt))
            X_new = np.core.records.fromarrays(Xt + s_arrays, names=['ts'] + s_names)
            return X_new, ye

    def _shuffle(self, X, y):
        ''' shuffles X and y '''
        ind = np.arange(len(y), dtype=np.int)
        np.random.shuffle(ind)
        X = X[ind]
        y = y[ind]
        return X, y


