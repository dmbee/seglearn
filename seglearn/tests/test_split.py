# Author: David Burns
# License: BSD

from numpy.random import rand
import numpy as np

from seglearn.split import TemporalKFold, temporal_split
from seglearn.base import TS_Data


def test_temporal_split():
    # test with length 1 series
    X = [rand(100, 10)]
    y = [5]
    Xtr, Xte, ytr, yte = temporal_split(X, y)
    check_split(X, Xtr, Xte, y, ytr, yte)

    X = [rand(100, 10)]
    y = [rand(100)]
    Xtr, Xte, ytr, yte = temporal_split(X, y)
    check_split(X, Xtr, Xte, y, ytr, yte)

    Xt = [rand(100, 10)]
    Xc = [5]
    X = TS_Data(Xt, Xc)
    y = [rand(100)]
    Xtr, Xte, ytr, yte = temporal_split(X, y)
    check_split(X, Xtr, Xte, y, ytr, yte)

    # test with lots of series
    Ns = 5
    X = np.array([rand(100, 10)] * Ns)
    y = rand(Ns)
    Xtr, Xte, ytr, yte = temporal_split(X, y)
    check_split(X, Xtr, Xte, y, ytr, yte)

    X = np.array([rand(100, 10)] * Ns)
    y = np.array([rand(100)] * Ns)
    Xtr, Xte, ytr, yte = temporal_split(X, y)
    check_split(X, Xtr, Xte, y, ytr, yte)

    Xt = np.array([rand(100, 10)] * Ns)
    Xc = rand(Ns)
    X = TS_Data(Xt, Xc)
    y = np.arange(Ns)
    Xtr, Xte, ytr, yte = temporal_split(X, y)
    check_split(X, Xtr, Xte, y, ytr, yte)


def test_temporal_k_fold():
    # test length 1 series
    splitter = TemporalKFold()
    X = [rand(100, 10)]
    y = [5]
    Xs, ys, cv = splitter.split(X, y)
    check_folds(Xs, ys, cv)

    X = [rand(100, 10)]
    y = [rand(100)]
    Xs, ys, cv = splitter.split(X, y)
    check_folds(Xs, ys, cv)

    Xt = [rand(100, 10)]
    Xc = [5]
    X = TS_Data(Xt, Xc)
    y = [rand(100)]
    Xs, ys, cv = splitter.split(X, y)
    check_folds(Xs, ys, cv)

    # test with lots of series
    splitter = TemporalKFold()
    Ns = 5
    X = np.array([rand(100, 10)] * Ns)
    y = rand(Ns)
    Xs, ys, cv = splitter.split(X, y)
    check_folds(Xs, ys, cv)

    X = np.array([rand(100, 10)] * Ns)
    y = np.array([rand(100)] * Ns)
    Xs, ys, cv = splitter.split(X, y)
    check_folds(Xs, ys, cv)

    Xt = np.array([rand(100, 10)] * Ns)
    Xc = rand(Ns)
    X = TS_Data(Xt, Xc)
    y = np.array([rand(100)] * Ns)
    Xs, ys, cv = splitter.split(X, y)
    check_folds(Xs, ys, cv)

    Xt = np.array([rand(100, 10)] * Ns)
    Xc = rand(Ns)
    X = TS_Data(Xt, Xc)
    y = rand(Ns)
    Xs, ys, cv = splitter.split(X, y)
    check_folds(Xs, ys, cv)


def check_ts_var(X, Xtr, Xte):
    assert np.all([np.array_equal(np.concatenate((Xtr[i], Xte[i])), X[i]) for i in range(len(X))])


def check_static_var(y, ytr, yte):
    assert np.array_equal(np.array(y), np.array(ytr))
    assert np.array_equal(np.array(y), np.array(yte))


def check_split(X, Xtr, Xte, y, ytr, yte):
    assert len(Xtr) == len(ytr)
    assert len(Xte) == len(yte)

    if isinstance(X, TS_Data):
        assert isinstance(Xtr, TS_Data)
        assert isinstance(Xte, TS_Data)
        Xt = X.ts_data
        Xtrt = Xtr.ts_data
        Xtet = Xte.ts_data
        Xc = X.context_data
        Xtrc = Xtr.context_data
        Xtec = Xte.context_data
        check_static_var(Xc, Xtrc, Xtec)
        check_ts_var(Xt, Xtrt, Xtet)
    else:
        check_ts_var(X, Xtr, Xte)

    if len(np.atleast_1d(y[0])) > 1:
        check_ts_var(y, ytr, yte)
    else:
        check_static_var(y, ytr, yte)


def check_folds(Xs, ys, cv):
    idj = []
    for i in range(len(cv)):
        assert len(Xs[cv[i][0]]) == len(ys[cv[i][0]])
        assert len(Xs[cv[i][1]]) == len(ys[cv[i][1]])
        idi = np.concatenate((cv[i][0], cv[i][1]))
        assert np.array_equal(np.sort(idi), np.arange(len(idi)))  # checks each value in fold
        idj.append(cv[i][1])
    idj = np.concatenate(idj)
    assert np.array_equal(np.sort(idj), np.arange(len(idj)))  # checks each value tested once
