import numpy as np
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline

# make an option for est to be none

class SegPipe(_BaseComposition):
    def __init__(self, feed, est, shuffle = True):
        self.feed = feed
        self.est = est
        self.shuffle = shuffle

    def fit(self, X, y, **feed_fit_params):
        ''' X can be array like time series or a dictionary with time and heuristic data '''
        self._reset()
        self._check_data(X)

        self.feed.fit(X, y, **feed_fit_params)
        X = self.feed.transform(X)

        if type(self.feed) is Pipeline:
            self.f_labels = self.feed._final_estimator.f_labels
        else:
            self.f_labels = self.feed.f_labels

        X, y = self._ts_expand(X, y)
        self.N_train = len(X)
        if self.shuffle is True:
            X, y = self._shuffle(X, y)
        self.est.fit(X, y)

    def transform(self, X, y = None):
        check_is_fitted(self, 'N_train')
        self._check_data(X)
        X = self.feed.transform(X)
        X, y = self._ts_expand(X, y)
        return self.est.transform(X), y


    def predict(self, X, y, **feed_fit_params):
        check_is_fitted(self, 'N_train')

        self._check_data(X)
        X = self.feed.transform(X)
        X, y = self._ts_expand(X, y)
        yp = self.est.predict(X)
        self.N_test = len(y)
        return y, yp


    def score(self, X, y, sample_weight = None):
        check_is_fitted(self, 'N_train')

        self._check_data(X)
        X = self.feed.transform(X)
        X, y = self._ts_expand(X, y)
        self.N_test = len(y)

        if sample_weight is None:
            return self.est.score(X, y)
        else:
            return self.est.score(X, y, sample_weight)

    def set_params(self, **params):

        # multiple elements in the pipe mapped to the same thing
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
    #     feed_params = self.feed.get_params()
    #     feed_params = {'f$'+key:feed_params[key] for key in feed_params}
    #
    #     est_params = self.est.get_params()
    #     est_params = {'e$' + key: est_params[key] for key in est_params}
    #
    #     return dict(list(feed_params.items()) + list(est_params.items()))


    def _reset(self):
        if hasattr(self,'N_train'):
            del self.N_train
            del self.f_labels
        if hasattr(self, 'N_test'):
            del self.N_test

    def _check_data(self, X, y = None): #todo: check y
        try:
            dnames = X.dtype.names
            assert 'ts' in dnames
        except:
            pass

    def _ts_expand(self, X, y = None):
        # if its a record array we need to expand 'ts' and 'h'
        try:
            dnames = X.dtype.names
        except:
            dnames = None

        if dnames is not None:
            Xe, Nt = self._expand_recarray(X)
        else:
            Xe, Nt = self._expand_array(X,y)

        ye = []
        if y is not None:
            N = len(y)
            for i in np.arange(N):
                ye.append(np.full(Nt[i], y[i]))
            ye = np.concatenate(ye)
        return Xe, ye

    def _expand_recarray(self, X):
        N = len(X)
        Nt = [len(X['ts'][i]) for i in np.arange(N)]
        Xt = np.concatenate(X['ts'])

        h_names = [h for h in X.dtype.names if h != 'ts']
        h_arrays = []
        for h in h_names:
            hi = []
            for i in np.arange(N):
                hi.append(np.full(Nt[i],X[h][i]))
            hi = np.concatenate(hi)
            h_arrays.append(hi)

        X_new = np.core.records.fromarrays(Xt + h_arrays, names=['ts'] + h_names)
        return X_new, Nt

    def _expand_array(self, X, y):
        N = len(X)
        Nt = [len(X[i]) for i in np.arange(N)]
        Xe = np.concatenate(X)
        return Xe, Nt

    def _shuffle(self, X, y):
        ind = np.arange(len(y))
        np.random.shuffle(ind)
        X, y = X[ind], y[ind]
        return X, y


