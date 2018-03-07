'''
This module has utilities for time series data input checking
'''
# Author: David Burns
# License: BSD

import numpy as np

def check_ts_data(X):
    '''
    Checks time series data is appropriate

    Parameters
    ----------
    X :

    Returns
    -------

    '''
    if type(X) is np.recarray:
        _check_ts(X['ts'])
    elif type(X) is list or type(X) is np.ndarray:
        _check_ts(X)
    else:
        raise TypeError

def _check_ts(X):
    Nvars = np.array([np.row_stack(X[i]).shape[1] for i in range(len(X))])
    assert len(np.unique(Nvars)) == 1

def ts_stats(X, y, fs = 1.0, class_labels = None):
    check_ts_data(X)
    try:
        dnames = X.dtype.names
    except:
        X = np.array(X)
        dnames = None
        H = 0

    if dnames is not None:
        X = X['ts']
        H = len([h for h in X.dtype.names if h != 'ts'])
    else:
        H = 0

    C = np.max(y) + 1 # number of classes
    if class_labels is None:
        class_labels = np.arange(C)

    N = len(X)
    if X[0].ndim > 1:
        D = X[0].shape[1]
    else:
        D = 1

    Ti = np.array([X[i].shape[0] for i in range(N)], dtype=np.float64) / fs
    ic = np.array([y == i for i in range(C)])
    Tic = [Ti[ic[i]] for i in range(C)]

    T = np.sum(Ti)

    total = {"N_series": N, "N_classes": C, "N_TS_vars": D, "N_H_vars": H, "Total_time": T,
              "Series_Time_Mean": np.mean(Ti),
              "Series_Time_Std": np.std(Ti),
              "Series_Time_Range": (np.min(Ti), np.max(Ti))}

    by_class = {"Class_labels": class_labels,
                "N_series": np.array([len(Tic[i]) for i in range(C)]),
                "Total_Time": np.array([np.sum(Tic[i]) for i in range(C)]),
                "Series_Time_Mean": np.array([np.mean(Tic[i]) for i in range(C)]),
                "Series_Time_Std": np.array([np.std(Tic[i]) for i in range(C)]),
                "Series_Time_Min": np.array([np.min(Tic[i]) for i in range(C)]),
                "Series_Time_Max": np.array([np.max(Tic[i]) for i in range(C)])}

    results = {'total': total,
               'by_class': by_class}

    return results















