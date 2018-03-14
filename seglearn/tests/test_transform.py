import seglearn.transform as transform
from seglearn.feature_functions import all_features, mean
from seglearn.util import make_ts_data, get_ts_data_parts

import numpy as np


def test_sliding_window():
    N = 1000
    width = 10
    ts = np.ones(N)
    for step in (1+np.arange(width)):
        sts = transform.sliding_window(ts, width, step)
        assert sts.shape[1] == width
        Nsts = 1 + (N - width) // step
        assert Nsts == sts.shape[0]


def test_sliding_tensor():
    N = 1000
    V = 5
    width = 10
    ts = np.ones((N, V))
    for step in (1 + np.arange(width)):
        sts = transform.sliding_tensor(ts, width, step)
        assert sts.shape[1] == width
        assert sts.shape[2] == V
        Nsts = 1 + (N - width) // step
        assert Nsts == sts.shape[0]

def test_feature_rep():
    # multivariate ts
    frep = transform.FeatureRep(features=all_features())
    X = np.random.rand(100, 10, 5)
    y = np.ones(100)
    frep.fit(X, y)
    Xt = frep.transform(X)
    assert Xt.shape[0] == len(X)
    assert len(frep.f_labels) == Xt.shape[1]

    # univariate ts
    X = np.random.rand(100, 10)
    y = np.ones(100)
    frep.fit(X, y)
    Xt = frep.transform(X)
    assert Xt.shape[0] == len(X)
    assert len(frep.f_labels) == Xt.shape[1]

    # single feature
    frep = transform.FeatureRep(features={'mean': mean})
    frep.fit(X, y)
    Xt = frep.transform(X)
    assert Xt.shape[0] == len(X)
    assert len(frep.f_labels) == Xt.shape[1]
    assert Xt.shape[1] == 1

    # ts with multivariate contextual data
    frep = transform.FeatureRep(features=all_features())
    X = make_ts_data(np.random.rand(100, 10, 5), np.random.rand(100,3))
    y = np.ones(100)
    frep.fit(X, y)
    Xt = frep.transform(X)
    assert Xt.shape[0] == len(X)
    assert len(frep.f_labels) == Xt.shape[1]

    # ts with univariate contextual data
    X = make_ts_data(np.random.rand(100, 10, 5), np.random.rand(100))
    y = np.ones(100)
    frep.fit(X, y)
    Xt = frep.transform(X)
    assert Xt.shape[0] == len(X)
    assert len(frep.f_labels) == Xt.shape[1]


def test_segmentx():
    width = 5
    vars = 5
    seg = transform.SegmentX(width = width)

    # multivariate ts data without context data
    Xt = [np.random.rand(100, vars), np.random.rand(100, vars), np.random.rand(100, vars)]
    y = np.random.rand(3)
    X = make_ts_data(Xt)
    seg.fit(X,y)
    Xs, ys, _ = seg.transform(X, y)
    N = len(ys)
    assert Xs.shape == (N, width, vars)

    # univariate ts data without context
    Xt = [np.random.rand(100), np.random.rand(100), np.random.rand(100)]
    y = np.random.rand(3)
    X = make_ts_data(Xt)
    seg.fit(X, y)
    Xs, ys, _ = seg.transform(X, y)
    N = len(ys)
    assert Xs.shape == (N, width)

    # multivariate ts data with context data
    Xt = [np.random.rand(100, vars), np.random.rand(200, vars), np.random.rand(50, vars)]
    Xc = np.random.rand(3, 4)
    y = np.random.rand(3)
    X = make_ts_data(Xt, Xc)
    seg.fit(X, y)
    Xs, ys, _ = seg.transform(X, y)
    Xst, Xsc = get_ts_data_parts(Xs)
    N = len(ys)
    assert Xst.shape == (N, width, vars)
    assert Xsc.shape == (N, 4)

    # ts data with univariate context data
    Xt = [np.random.rand(100), np.random.rand(200), np.random.rand(50)]
    Xc = np.random.rand(3)
    y = np.random.rand(3)
    X = make_ts_data(Xt, Xc)
    seg.fit(X, y)
    Xs, ys, _ = seg.transform(X, y)
    Xst, Xsc = get_ts_data_parts(Xs)
    N = len(ys)
    assert Xst.shape == (N, width)
    assert Xsc.shape == (N, 1)

    # same number as context vars and time vars
    # this would cause broadcasting failure before implementation of TS_Data class
    Xt = [np.random.rand(100, vars), np.random.rand(200, vars), np.random.rand(50, vars)]
    Xc = np.random.rand(3, vars)
    y = np.random.rand(3)
    X = make_ts_data(Xt, Xc)
    seg.fit(X, y)
    Xs, ys, _ = seg.transform(X, y)
    Xst, Xsc = get_ts_data_parts(Xs)
    N = len(ys)
    assert Xst.shape == (N, width, vars)
    assert Xsc.shape == (N, 5)


def test_segmentxy():
    Nt = 100
    width = 5
    vars = 5
    seg = transform.SegmentXY(width = width)

    # multivariate ts data without context data
    Xt = [np.random.rand(Nt, vars), np.random.rand(Nt, vars), np.random.rand(Nt, vars)]
    y = [np.random.rand(Nt), np.random.rand(Nt), np.random.rand(Nt)]
    X = make_ts_data(Xt)
    seg.fit(X, y)
    Xs, ys, _ = seg.transform(X, y)
    N = len(ys)
    assert Xs.shape == (N, width, vars)

    # univariate ts data without context data
    Xt = [np.random.rand(Nt), np.random.rand(2*Nt), np.random.rand(3*Nt)]
    y = [np.random.rand(Nt), np.random.rand(2*Nt), np.random.rand(3*Nt)]
    X = make_ts_data(Xt)
    seg.fit(X, y)
    Xs, ys, _ = seg.transform(X, y)
    N = len(ys)
    assert Xs.shape == (N, width)

    # multivariate ts data with context data
    Xt = [np.random.rand(Nt, vars), np.random.rand(2*Nt, vars), np.random.rand(Nt, vars)]
    Xc = np.random.rand(3, 4)
    y = [np.random.rand(Nt), np.random.rand(2*Nt), np.random.rand(Nt)]
    X = make_ts_data(Xt, Xc)
    seg.fit(X, y)
    Xs, ys, _ = seg.transform(X, y)
    Xst, Xsc = get_ts_data_parts(Xs)
    N = len(ys)
    assert Xst.shape == (N, width, vars)
    assert Xsc.shape == (N, 4)

    # ts data with univariate context data
    Xt = [np.random.rand(Nt, vars), np.random.rand(2 * Nt, vars), np.random.rand(Nt, vars)]
    Xc = np.random.rand(3)
    y = [np.random.rand(Nt), np.random.rand(2 * Nt), np.random.rand(Nt)]
    X = make_ts_data(Xt, Xc)
    seg.fit(X, y)
    Xs, ys, _ = seg.transform(X, y)
    Xst, Xsc = get_ts_data_parts(Xs)
    N = len(ys)
    assert Xst.shape == (N, width, vars)
    assert Xsc.shape == (N, 1)


    # same number as context vars and time vars
    # this would cause broadcasting failure before implementation of TS_Data class
    Xt = [np.random.rand(Nt, vars), np.random.rand(2*Nt, vars), np.random.rand(Nt, vars)]
    Xc = np.random.rand(3, vars)
    y = [np.random.rand(Nt), np.random.rand(2 * Nt), np.random.rand(Nt)]
    X = make_ts_data(Xt, Xc)
    seg.fit(X, y)
    Xs, ys, _ = seg.transform(X, y)
    Xst, Xsc = get_ts_data_parts(Xs)
    N = len(ys)
    assert Xst.shape == (N, width, vars)
    assert Xsc.shape == (N, 5)

    # lets do a forecast test
    seg = transform.SegmentXY(width=width, forecast=5)
    Xt = [np.random.rand(Nt, vars), np.random.rand(2 * Nt, vars), np.random.rand(Nt, vars)]
    Xc = np.random.rand(3, 4)
    y = [np.random.rand(Nt), np.random.rand(2 * Nt), np.random.rand(Nt)]
    X = make_ts_data(Xt, Xc)
    seg.fit(X, y)
    Xs, ys, _ = seg.transform(X, y)
    Xst, Xsc = get_ts_data_parts(Xs)
    N = len(ys)
    assert Xst.shape == (N, width, vars)
    assert Xsc.shape == (N, 4)
