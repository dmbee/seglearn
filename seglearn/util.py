"""
This module has utilities for time series data input checking
"""
# Author: David Burns
# License: BSD

from pandas import DataFrame
import numpy as np
import warnings

from seglearn.base import TS_Data

__all__ = ['get_ts_data_parts', 'check_ts_data', 'check_ts_data_with_ts_target', 'ts_stats']


def get_ts_data_parts(X):
    """
    Separates time series data object into time series variables and contextual variables

    Parameters
    ----------
    X : array-like, shape [n_series, ...]
       Time series data and (optionally) contextual data

    Returns
    -------
    Xt : array-like, shape [n_series, ]
        Time series data
    Xs : array-like, shape [n_series, n_contextd = np.colum _variables]
        contextual variables

    """

    if isinstance(X, TS_Data):
        return X.ts_data, X.context_data
    elif isinstance(X, DataFrame):
        return X.ts_data.values, X.drop(columns=['ts_data']).values
    else:
        return X, None

def check_ts_data(X, y=None):
    """
    Checks time series data is good. If not raises value error.

    Parameters
    ----------
    X : array-like, shape [n_series, ...]
       Time series data and (optionally) contextual data
       
    Returns
    -------
    ts_target : bool
        target (y) is a time series      
        
    """

    if y is not None:
        Nx = len(X)
        Ny = len(y)

        if Nx != Ny:
            raise ValueError("Number of time series different in X (%d) and y (%d)"
                             % (Nx, Ny))

        Xt, _ = get_ts_data_parts(X)
        Ntx = np.array([len(Xt[i]) for i in np.arange(Nx)])
        Nty = np.array([len(np.atleast_1d(y[i])) for i in np.arange(Nx)])

        if np.count_nonzero(Nty == 1) == Nx:  # all targets are single values
            return False
        elif np.count_nonzero(Nty == Ntx) == Nx:  # y is a time series
            return True
        elif np.count_nonzero(Nty == Nty[0]) == Nx:  # target vector (eg multilabel or onehot)
            return False
        else:
            raise ValueError("Invalid time series lengths.\n"
                             "Ns: ", Nx,
                             "Ntx: ", Ntx,
                             "Nty: ", Nty)


def check_ts_data_with_ts_target(X, y=None):
    """
    Checks time series data with time series target is good. If not raises value error.

    Parameters
    ----------
    X : array-like, shape [n_series, ...]
       Time series data and (optionally) contextual data
    y : array-like, shape [n_series, ...]
        target data
    """
    if y is not None:
        Nx = len(X)
        Ny = len(y)

        if Nx != Ny:
            raise ValueError("Number of time series different in X (%d) and y (%d)"
                             % (Nx, Ny))

        Xt, _ = get_ts_data_parts(X)
        Ntx = np.array([len(Xt[i]) for i in np.arange(Nx)])
        Nty = np.array([len(np.atleast_1d(y[i])) for i in np.arange(Nx)])

        if np.count_nonzero(Nty == Ntx) == Nx:
            return
        else:
            raise ValueError("Invalid time series lengths.\n"
                             "Ns: ", Nx,
                             "Ntx: ", Ntx,
                             "Nty: ", Nty)


def ts_stats(Xt, y, fs=1.0, class_labels=None):
    """
    Generates some helpful statistics about the data X

    Parameters
    ----------
    X : array-like, shape [n_series, ...]
       Time series data and (optionally) contextual data
    y : array-like, shape [n_series]
        target data
    fs : float
        sampling frequency
    class_labels : list of strings, default None
        List of target class names


    Returns
    -------
    results : dict
        | Dictionary of relevant statistics for the time series data
        | results['total'] has stats for the whole data set
        | results['by_class'] has stats segragated by target class

    """
    check_ts_data(Xt)
    Xt, Xs = get_ts_data_parts(Xt)

    if Xs is not None:
        S = len(np.atleast_1d(Xs[0]))
    else:
        S = 0

    C = np.max(y) + 1  # number of classes
    if class_labels is None:
        class_labels = np.arange(C)

    N = len(Xt)
    if Xt[0].ndim > 1:
        D = Xt[0].shape[1]
    else:
        D = 1

    Ti = np.array([Xt[i].shape[0] for i in range(N)], dtype=np.float64) / fs
    ic = np.array([y == i for i in range(C)])
    Tic = [Ti[ic[i]] for i in range(C)]

    T = np.sum(Ti)

    total = {"n_series": N, "n_classes": C, "n_TS_vars": D, "n_context_vars": S, "Total_Time": T,
             "Series_Time_Mean": np.mean(Ti),
             "Series_Time_Std": np.std(Ti),
             "Series_Time_Range": (np.min(Ti), np.max(Ti))}

    by_class = {"Class_labels": class_labels,
                "n_series": np.array([len(Tic[i]) for i in range(C)]),
                "Total_Time": np.array([np.sum(Tic[i]) for i in range(C)]),
                "Series_Time_Mean": np.array([np.mean(Tic[i]) for i in range(C)]),
                "Series_Time_Std": np.array([np.std(Tic[i]) for i in range(C)]),
                "Series_Time_Min": np.array([np.min(Tic[i]) for i in range(C)]),
                "Series_Time_Max": np.array([np.max(Tic[i]) for i in range(C)])}

    results = {'total': total,
               'by_class': by_class}

    return results

def interp_sort(t, x):
    """
    sorts time series x by timestamp t, removing duplicates in the first entry

    this is required to user the scipy interp1d methods which returns nan when there are duplicate
    values for t_min

    this can be removed once the scipy issue is fixed

    Parameters
    ----------
    t : array-like, shape [n]
        timestamps
    x : array-like, shape [n, ]
        data

    Returns
    -------
    t : array-like, shape [n]
        timestamps
    x : array-like, shape [n, ]
        data
    """
    if len(t) != len(x):
        raise ValueError("Interpolation time and value errors not equal")

    ind = np.argsort(t)
    t = t[ind]
    x = x[ind]

    t, ind = np.unique(t, return_index=True)

    if len(t) < len(x):
        warnings.warn("Interpolation time has duplicate time indices", UserWarning)
        x = x[ind]

    return t, x


def segmented_prediction_to_series(yp, step, width, categorical_target=False):
    """
    resamples prediction on a single segmented series to original series sampling

    Parameters
    ----------
    yp : array-like, shape [n, ]
        prediction on segmented series
    step : int
        segmentation step size (number of samples)
    width : int
        segmentation width (number of samples)
    categorical_target : boolean
        set to True for classification problems and False for regression problems

    Returns
    -------
    yt : array-like, shape [n, ]
        resampled prediction

    """
    # average regression predictions if highly overlapping
    if not categorical_target and step < 0.5 * width:
        mask = segmentation_mask(len(yp), step, width)
        counts = np.bincount(mask)
        yt = np.repeat(yp, width, axis=0)

        if yt.ndim == 1:
            yt = np.nan_to_num(np.bincount(mask, weights=yt) / counts)
        else:
            yt = np.column_stack(
                [np.nan_to_num(np.bincount(mask, weights=yt[:, i]) / counts)
                 for i in range(yt.shape[1])]
            )
        return yt
    else:
        yt = np.repeat(yp[0:-1], step, axis=0)
        ye = np.repeat(yp[-1:], width, axis=0)
        yt = np.append(yt, ye, axis=0)
        return yt


def segmentation_mask(N, step, width):
    mask = np.tile(np.arange(width), (N, 1))
    steps = np.array([np.arange(start=0, stop=N * step, step=step)]).transpose()
    steps = np.tile(steps, (1, width))
    mask = mask + steps
    return mask.flatten()
