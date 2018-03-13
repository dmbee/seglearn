from seglearn.split import TemporalKFold, temporal_split
from seglearn.base import TS_Data

from numpy.random import rand
import numpy as np

def test_temporal_split():
    # test with length 1 series
    X = [rand(100,10)]
    y = [5]
    Xtr, Xte, ytr, yte = temporal_split(X, y)
    assert len(Xtr) == len(ytr)
    assert len(Xte) == len(yte)

    X = [rand(100, 10)]
    y = [rand(100)]
    Xtr, Xte, ytr, yte = temporal_split(X, y)
    assert len(Xtr) == len(ytr)
    assert len(Xte) == len(yte)

    Xt = [rand(100, 10)]
    Xc = [5]
    X = TS_Data(Xt, Xc)
    y = [rand(100)]
    Xtr, Xte, ytr, yte = temporal_split(X, y)
    assert len(Xtr) == len(ytr)
    assert len(Xte) == len(yte)

    # test with lots of series
    Ns = 5
    X = np.array([rand(100,10) for i in range(Ns)])
    y = rand(Ns)
    Xtr, Xte, ytr, yte = temporal_split(X, y)
    assert len(Xtr) == len(ytr)
    assert len(Xte) == len(yte)

    X = np.array([rand(100,10) for i in range(Ns)])
    y = np.array([rand(100) for i in range(Ns)])
    Xtr, Xte, ytr, yte = temporal_split(X, y)
    assert len(Xtr) == len(ytr)
    assert len(Xte) == len(yte)


    Xt = np.array([rand(100,10) for i in range(Ns)])
    Xc = rand(Ns)
    X = TS_Data(Xt,Xc)
    y = np.arange(Ns)
    Xtr, Xte, ytr, yte = temporal_split(X, y)
    assert len(Xtr) == len(ytr)
    assert len(Xte) == len(yte)


def test_temporal_k_fold():
    # test length 1 series
    splitter = TemporalKFold()
    X = [rand(100,10)]
    y = [5]
    X, y, cv = splitter.split(X, y)
    check_folds(X, y, cv)

    X = [rand(100,10)]
    y = [rand(100)]
    X, y, cv = splitter.split(X, y)
    check_folds(X, y, cv)

    Xt = [rand(100,10)]
    Xc = [5]
    X = TS_Data(Xt, Xc)
    y = [rand(100)]
    X, y, cv = splitter.split(X, y)
    check_folds(X, y, cv)

    # test with lots of series
    Ns = 5
    X = np.array([rand(100, 10) for i in range(Ns)])
    y = rand(Ns)
    X, y, cv = splitter.split(X, y)
    check_folds(X, y, cv)

    X = np.array([rand(100, 10) for i in range(Ns)])
    y = np.array([rand(100) for i in range(Ns)])
    X, y, cv = splitter.split(X, y)
    check_folds(X, y, cv)

    Xt = np.array([rand(100, 10) for i in range(Ns)])
    Xc = rand(Ns)
    X = TS_Data(Xt,Xc)
    y = np.array([rand(100) for i in range(Ns)])
    X, y, cv = splitter.split(X, y)
    check_folds(X, y, cv)

    Xt = np.array([rand(100, 10) for i in range(Ns)])
    Xc = rand(Ns)
    X = TS_Data(Xt,Xc)
    y = rand(Ns)
    X, y, cv = splitter.split(X, y)
    check_folds(X, y, cv)


def check_folds(X, y, cv):
    for i in range(len(cv)):
        assert len(X[cv[i][0]]) == len(y[cv[i][0]])
        assert len(X[cv[i][1]]) == len(y[cv[i][1]])