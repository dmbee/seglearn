# Author: David Burns
# License: BSD

import numpy as np

import seglearn.transform as transform
from seglearn.base import TS_Data
from seglearn.feature_functions import all_features, mean
from seglearn.util import get_ts_data_parts


def test_sliding_window():
    N = 1000
    width = 10
    ts = np.random.rand(N)
    for step in 1 + np.arange(width):
        sts = transform.sliding_window(ts, width, step)
        assert sts.shape[1] == width
        Nsts = 1 + (N - width) // step
        assert Nsts == sts.shape[0]
        assert np.all(np.isin(sts, ts))

        # reconstruct the ts
        if step == 1:
            assert np.array_equal(np.concatenate((sts[:, 0], sts[-1, 1:width])), ts)

        if step == width:
            assert np.array_equal(sts.ravel(), ts)


def test_sliding_tensor():
    N = 1000
    V = 5
    width = 10
    ts = np.random.rand(N, V)
    for step in 1 + np.arange(width):
        sts = transform.sliding_tensor(ts, width, step)
        assert sts.shape[1] == width
        assert sts.shape[2] == V
        Nsts = 1 + (N - width) // step
        assert Nsts == sts.shape[0]
        for j in range(V):
            assert np.all(np.isin(sts[:, :, j], ts[:, j]))

        # todo: reconstruct tensor ts


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
    X = TS_Data(np.random.rand(100, 10, 5), np.random.rand(100, 3))
    y = np.ones(100)
    frep.fit(X, y)
    Xt = frep.transform(X)
    assert Xt.shape[0] == len(X)
    assert len(frep.f_labels) == Xt.shape[1]

    # ts with univariate contextual data
    X = TS_Data(np.random.rand(100, 10, 5), np.random.rand(100))
    y = np.ones(100)
    frep.fit(X, y)
    Xt = frep.transform(X)
    assert Xt.shape[0] == len(X)
    assert len(frep.f_labels) == Xt.shape[1]


def test_segmentx():
    width = 5
    nvars = 5
    seg = transform.SegmentX(width=width)

    # multivariate ts data without context data
    X = [np.random.rand(100, nvars), np.random.rand(100, nvars), np.random.rand(100, nvars)]
    y = np.random.rand(3)
    seg.fit(X, y)
    Xs, ys, _ = seg.transform(X, y)
    N = len(ys)
    assert Xs.shape == (N, width, nvars)

    # univariate ts data without context
    X = [np.random.rand(100), np.random.rand(100), np.random.rand(100)]
    y = np.random.rand(3)
    seg.fit(X, y)
    Xs, ys, _ = seg.transform(X, y)
    N = len(ys)
    assert Xs.shape == (N, width)

    # multivariate ts data with context data
    Xt = [np.random.rand(100, nvars), np.random.rand(200, nvars), np.random.rand(50, nvars)]
    Xc = np.random.rand(3, 4)
    y = np.random.rand(3)
    X = TS_Data(Xt, Xc)
    seg.fit(X, y)
    Xs, ys, _ = seg.transform(X, y)
    Xst, Xsc = get_ts_data_parts(Xs)
    N = len(ys)
    assert Xst.shape == (N, width, nvars)
    assert Xsc.shape == (N, 4)

    # ts data with univariate context data
    Xt = [np.random.rand(100), np.random.rand(200), np.random.rand(50)]
    Xc = np.random.rand(3)
    y = np.random.rand(3)
    X = TS_Data(Xt, Xc)
    seg.fit(X, y)
    Xs, ys, _ = seg.transform(X, y)
    Xst, Xsc = get_ts_data_parts(Xs)
    N = len(ys)
    assert Xst.shape == (N, width)
    assert Xsc.shape == (N, 1)

    # same number as context vars and time vars
    # this would cause broadcasting failure before implementation of TS_Data class
    Xt = [np.random.rand(100, nvars), np.random.rand(200, nvars), np.random.rand(50, nvars)]
    Xc = np.random.rand(3, nvars)
    y = np.random.rand(3)
    X = TS_Data(Xt, Xc)
    seg.fit(X, y)
    Xs, ys, _ = seg.transform(X, y)
    Xst, Xsc = get_ts_data_parts(Xs)
    N = len(ys)
    assert Xst.shape == (N, width, nvars)
    assert Xsc.shape == (N, 5)


def test_segmentxy():
    Nt = 100
    width = 5
    nvars = 5
    seg = transform.SegmentXY(width=width)

    # multivariate ts data without context data
    X = [np.random.rand(Nt, nvars), np.random.rand(Nt, nvars), np.random.rand(Nt, nvars)]
    y = [np.random.rand(Nt), np.random.rand(Nt), np.random.rand(Nt)]
    seg.fit(X, y)
    Xs, ys, _ = seg.transform(X, y)
    N = len(ys)
    assert Xs.shape == (N, width, nvars)

    # univariate ts data without context data
    X = [np.random.rand(Nt), np.random.rand(2 * Nt), np.random.rand(3 * Nt)]
    y = [np.random.rand(Nt), np.random.rand(2 * Nt), np.random.rand(3 * Nt)]
    seg.fit(X, y)
    Xs, ys, _ = seg.transform(X, y)
    N = len(ys)
    assert Xs.shape == (N, width)

    # multivariate ts data with context data
    Xt = [np.random.rand(Nt, nvars), np.random.rand(2 * Nt, nvars), np.random.rand(Nt, nvars)]
    Xc = np.random.rand(3, 4)
    y = [np.random.rand(Nt), np.random.rand(2 * Nt), np.random.rand(Nt)]
    X = TS_Data(Xt, Xc)
    seg.fit(X, y)
    Xs, ys, _ = seg.transform(X, y)
    Xst, Xsc = get_ts_data_parts(Xs)
    N = len(ys)
    assert Xst.shape == (N, width, nvars)
    assert Xsc.shape == (N, 4)

    # ts data with univariate context data
    Xt = [np.random.rand(Nt, nvars), np.random.rand(2 * Nt, nvars), np.random.rand(Nt, nvars)]
    Xc = np.random.rand(3)
    y = [np.random.rand(Nt), np.random.rand(2 * Nt), np.random.rand(Nt)]
    X = TS_Data(Xt, Xc)
    seg.fit(X, y)
    Xs, ys, _ = seg.transform(X, y)
    Xst, Xsc = get_ts_data_parts(Xs)
    N = len(ys)
    assert Xst.shape == (N, width, nvars)
    assert Xsc.shape == (N, 1)

    # same number as context vars and time vars
    # this would cause broadcasting failure before implementation of TS_Data class
    Xt = [np.random.rand(Nt, nvars), np.random.rand(2 * Nt, nvars), np.random.rand(Nt, nvars)]
    Xc = np.random.rand(3, nvars)
    y = [np.random.rand(Nt), np.random.rand(2 * Nt), np.random.rand(Nt)]
    X = TS_Data(Xt, Xc)
    seg.fit(X, y)
    Xs, ys, _ = seg.transform(X, y)
    Xst, Xsc = get_ts_data_parts(Xs)
    N = len(ys)
    assert Xst.shape == (N, width, nvars)
    assert Xsc.shape == (N, 5)


def test_segmentxyforecast():
    Nt = 100
    width = 5
    nvars = 5

    # lets do a forecast test
    seg = transform.SegmentXYForecast(width=width, forecast=5)
    Xt = [np.random.rand(Nt, nvars), np.random.rand(2 * Nt, nvars), np.random.rand(Nt, nvars)]
    Xc = np.random.rand(3, 4)
    y = [np.random.rand(Nt), np.random.rand(2 * Nt), np.random.rand(Nt)]
    X = TS_Data(Xt, Xc)
    seg.fit(X, y)
    Xs, ys, _ = seg.transform(X, y)
    Xst, Xsc = get_ts_data_parts(Xs)
    N = len(ys)
    assert Xst.shape == (N, width, nvars)
    assert Xsc.shape == (N, 4)

    # univariate X
    nvars = 1
    seg = transform.SegmentXYForecast(width=width, forecast=5)
    X = [np.random.rand(Nt), np.random.rand(2 * Nt), np.random.rand(Nt)]
    y = [np.random.rand(Nt), np.random.rand(2 * Nt), np.random.rand(Nt)]
    seg.fit(X, y)
    Xs, ys, _ = seg.transform(X, y)
    Xst, Xsc = get_ts_data_parts(Xs)
    N = len(ys)
    assert Xst.shape == (N, width)


def test_pad_trunc():
    Nt = 100
    width = 5
    nvars = 5
    seg = transform.PadTrunc(width=width)

    # multivariate ts data without context data
    X = [np.random.rand(Nt, nvars), np.random.rand(Nt, nvars), np.random.rand(Nt, nvars)]
    y = [np.random.rand(Nt), np.random.rand(Nt), np.random.rand(Nt)]
    seg.fit(X, y)
    Xs, ys, _ = seg.transform(X, y)
    N = len(ys)
    assert Xs.shape == (N, width, nvars)
    assert np.all([np.equal(X[i][0:width], Xs[i]) for i in range(len(X))])
    assert np.all([np.equal(y[i][0:width], ys[i]) for i in range(len(y))])

    # univariate ts data without context data
    X = [np.random.rand(Nt), np.random.rand(2 * Nt), np.random.rand(3 * Nt)]
    y = [np.random.rand(Nt), np.random.rand(2 * Nt), np.random.rand(3 * Nt)]
    seg.fit(X, y)
    Xs, ys, _ = seg.transform(X, y)
    N = len(ys)
    assert Xs.shape == (N, width)
    assert np.all([np.equal(X[i][0:width], Xs[i]) for i in range(len(X))])
    assert np.all([np.equal(y[i][0:width], ys[i]) for i in range(len(y))])

    # multivariate ts data with context data
    Xt = [np.random.rand(Nt, nvars), np.random.rand(2 * Nt, nvars), np.random.rand(Nt, nvars)]
    Xc = np.random.rand(3, 4)
    y = [np.random.rand(Nt), np.random.rand(2 * Nt), np.random.rand(Nt)]
    X = TS_Data(Xt, Xc)
    seg.fit(X, y)
    Xs, ys, _ = seg.transform(X, y)
    Xst, Xsc = get_ts_data_parts(Xs)
    N = len(ys)
    assert Xst.shape == (N, width, nvars)
    assert Xsc.shape == (N, 4)
    assert np.all([np.equal(Xt[i][0:width], Xst[i]) for i in range(len(Xt))])
    assert np.all([np.equal(Xc[i], Xsc[i]) for i in range(len(Xt))])
    assert np.all([np.equal(y[i][0:width], ys[i]) for i in range(len(y))])

    # ts data with univariate context data
    Xt = [np.random.rand(Nt, nvars), np.random.rand(2 * Nt, nvars), np.random.rand(Nt, nvars)]
    Xc = np.random.rand(3)
    y = [np.random.rand(Nt), np.random.rand(2 * Nt), np.random.rand(Nt)]
    X = TS_Data(Xt, Xc)
    seg.fit(X, y)
    Xs, ys, _ = seg.transform(X, y)
    Xst, Xsc = get_ts_data_parts(Xs)
    N = len(ys)
    assert Xst.shape == (N, width, nvars)
    assert Xsc.shape == (N,)
    assert np.all([np.equal(Xt[i][0:width], Xst[i]) for i in range(len(Xt))])
    assert np.all([np.equal(Xc[i], Xsc[i]) for i in range(len(Xt))])
    assert np.all([np.equal(y[i][0:width], ys[i]) for i in range(len(y))])

    # same number as context vars and time vars
    # this would cause broadcasting failure before implementation of TS_Data class
    Xt = [np.random.rand(Nt, nvars), np.random.rand(2 * Nt, nvars), np.random.rand(Nt, nvars)]
    Xc = np.random.rand(3, nvars)
    y = [np.random.rand(Nt), np.random.rand(2 * Nt), np.random.rand(Nt)]
    X = TS_Data(Xt, Xc)
    seg.fit(X, y)
    Xs, ys, _ = seg.transform(X, y)
    Xst, Xsc = get_ts_data_parts(Xs)
    N = len(ys)
    assert Xst.shape == (N, width, nvars)
    assert Xsc.shape == (N, 5)
    assert np.all([np.equal(Xt[i][0:width], Xst[i]) for i in range(len(Xt))])
    assert np.all([np.equal(Xc[i], Xsc[i]) for i in range(len(Xt))])
    assert np.all([np.equal(y[i][0:width], ys[i]) for i in range(len(y))])

    width = 5
    nvars = 5
    seg = transform.PadTrunc(width=width)

    # multivariate ts data without context data
    X = [np.random.rand(100, nvars), np.random.rand(100, nvars), np.random.rand(100, nvars)]
    y = np.random.rand(3)
    seg.fit(X, y)
    Xs, ys, _ = seg.transform(X, y)
    N = len(ys)
    assert Xs.shape == (N, width, nvars)
    assert np.all([np.equal(X[i][0:width], Xs[i]) for i in range(len(Xt))])
    assert np.all([np.equal(y[i], ys[i]) for i in range(len(y))])

    # univariate ts data without context
    X = [np.random.rand(100), np.random.rand(100), np.random.rand(100)]
    y = np.random.rand(3)
    seg.fit(X, y)
    Xs, ys, _ = seg.transform(X, y)
    N = len(ys)
    assert Xs.shape == (N, width)
    assert np.all([np.equal(X[i][0:width], Xs[i]) for i in range(len(Xt))])
    assert np.all([np.equal(y[i], ys[i]) for i in range(len(y))])

    # multivariate ts data with context data
    Xt = [np.random.rand(100, nvars), np.random.rand(200, nvars), np.random.rand(50, nvars)]
    Xc = np.random.rand(3, 4)
    y = np.random.rand(3)
    X = TS_Data(Xt, Xc)
    seg.fit(X, y)
    Xs, ys, _ = seg.transform(X, y)
    Xst, Xsc = get_ts_data_parts(Xs)
    N = len(ys)
    assert Xst.shape == (N, width, nvars)
    assert Xsc.shape == (N, 4)
    assert np.all([np.equal(Xt[i][0:width], Xst[i]) for i in range(len(Xt))])
    assert np.all([np.equal(Xc[i], Xsc[i]) for i in range(len(Xt))])
    assert np.all([np.equal(y[i], ys[i]) for i in range(len(y))])


def test_interp():
    N = 100
    t = np.arange(N) + np.random.rand(N)
    X = [np.column_stack([t, np.random.rand(N)])]
    y = [np.random.rand(N)]

    interp = transform.Interp(2)
    interp.fit(X)
    Xc, yc, swt = interp.transform(X, y)

    assert len(Xc[0]) == N / 2
    assert len(yc[0]) == N / 2

    y = [np.random.randint(0, 5, N)]
    interp = transform.Interp(5, kind='cubic', categorical_target=True)
    interp.fit(X, y)
    Xc, yc, swt = interp.transform(X, y)

    assert len(Xc[0]) == N / 5
    assert len(yc[0]) == N / 5
    assert np.all(np.isin(yc, np.arange(6)))
