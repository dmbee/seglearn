#from .util import check_ts_input

import numpy as np

# from sklearn.model_selection._split import _BaseKFold

class TemporalKFold():

    def __init__(self, n_splits = 3, shuffle = False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = None


    def split(self, X, y, groups = None):
        #check_ts_input(X)
        N = len(X)
        if type(X) is np.recarray:
            X_new = np.concatenate([X for i in range(self.n_splits)])
            X_new['ts'] = self._ts_slice(X['ts'])
            N_new = len(X_new['ts'])
        else:
            X_new = self._ts_slice(X)
            N_new = len(X_new)

        y_new = np.concatenate([y for i in range(self.n_splits)])

        test = [np.full(N_new, False) for i in range(self.n_splits)]
        for i in range(self.n_splits):
            test[i][np.arange(N*i,N*(i+1))] = True
        train = [np.logical_not(test[i]) for i in range(self.n_splits)]

        test = [np.arange(N_new)[test[i]] for i in range(self.n_splits)]
        train = [np.arange(N_new)[train[i]] for i in range(self.n_splits)]

        cv = list(zip(train, test))

        return X_new, y_new, cv

    def _ts_slice(self, X):
        ''' takes a time series, splits each one into folds '''
        N = len(X)
        X_new = []
        for i in range(self.n_splits):
            for j in range(N):
                Njs = int(len(X[j]) / self.n_splits)
                X_new.append(X[j][(Njs*i):(Njs*(i+1))])

        return X_new








