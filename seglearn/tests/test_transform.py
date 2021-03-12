# Author: David Burns
# License: BSD

import warnings

import numpy as np
import pytest

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
        sts_c = transform.sliding_window(ts, width, step, 'C')
        assert sts.flags.f_contiguous and sts_c.flags.c_contiguous
        assert sts.shape[1] == width and sts_c.shape[1] == width
        Nsts = 1 + (N - width) // step
        assert Nsts == sts.shape[0] and Nsts == sts_c.shape[0]
        assert np.all(np.isin(sts, ts)) and np.all(np.isin(sts_c, ts))

        # reconstruct the ts
        if step == 1:
            assert np.array_equal(np.concatenate((sts[:, 0], sts[-1, 1:width])), ts)
            assert np.array_equal(np.concatenate((sts_c[:, 0], sts_c[-1, 1:width])), ts)

        if step == width:
            assert np.array_equal(sts.ravel(), ts)
            assert np.array_equal(sts_c.ravel(), ts)


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

    final_tensor = []
    for step in 1 + np.arange(width):
        sts = transform.sliding_tensor(ts, width, step, 'C')
        final_tensor.append(sts)
        assert sts.flags.c_contiguous
        assert sts.shape[1] == width
        assert sts.shape[2] == V
        Nsts = 1 + (N - width) // step
        assert Nsts == sts.shape[0]
        for j in range(V):
            assert np.all(np.isin(sts[:, :, j], ts[:, j]))
    assert np.concatenate(final_tensor).flags.c_contiguous

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


@pytest.mark.filterwarnings("ignore:deprecated, use Segment class")
def test_segmentx():
    # test illegal parameter settings
    with pytest.raises(ValueError):
        transform.SegmentX(width=0)                  # illegal width value
    with pytest.raises(ValueError):
        transform.SegmentX(overlap=None, step=None)  # either overlap or step must be defined
    with pytest.raises(ValueError):
        transform.SegmentX(overlap=-1, step=None)    # illegal overlap value
    with pytest.raises(ValueError):
        transform.SegmentX(step=0)                   # illegal step value
    with pytest.raises(ValueError):
        transform.SegmentX(order=None)               # illegal order

    # test _step property working as expected

    seg = transform.Segment(width=10, overlap=0.5)
    assert seg._step == 5

    # test precedence of step over overlap
    seg = transform.Segment(width=10, overlap=1, step=1)
    assert seg._step == 1

    # illegal overlap value, but valid step value
    seg = transform.Segment(overlap=-1, step=1)
    assert seg._step == 1

    # test shape of segmented data
    width = 5
    nvars = 5
    seg = transform.Segment(width=width)

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


@pytest.mark.filterwarnings("ignore:deprecated, use Segment class")
def test_segmentxy():
    # test illegal parameter settings
    with pytest.raises(ValueError):
        transform.SegmentXY(width=0)                  # illegal width value
    with pytest.raises(ValueError):
        transform.SegmentXY(overlap=None, step=None)  # either overlap or step must be defined
    with pytest.raises(ValueError):
        transform.SegmentXY(overlap=-1, step=None)    # illegal overlap value
    with pytest.raises(ValueError):
        transform.SegmentXY(step=0)                   # illegal step value
    with pytest.raises(ValueError):
        transform.SegmentXY(order=None)               # illegal order

    # test _step property working as expected
    seg = transform.SegmentXY(width=10, overlap=0.5)
    assert seg._step == 5

    # test precedence of step over overlap
    seg = transform.SegmentXY(width=10, overlap=1, step=1)
    assert seg._step == 1

    # illegal overlap value, but valid step value
    seg = transform.SegmentXY(overlap=-1, step=1)
    assert seg._step == 1

    # test shape of segmented data
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
    # test illegal parameter settings
    with pytest.raises(ValueError):
        transform.SegmentXYForecast(width=0)                  # illegal width value
    with pytest.raises(ValueError):
        transform.SegmentXYForecast(overlap=None, step=None)  # either overlap or step must be defined
    with pytest.raises(ValueError):
        transform.SegmentXYForecast(overlap=-1, step=None)    # illegal overlap value
    with pytest.raises(ValueError):
        transform.SegmentXYForecast(step=0)                   # illegal step value
    with pytest.raises(ValueError):
        transform.SegmentXYForecast(order=None)               # illegal order
    with pytest.raises(ValueError):
        transform.SegmentXYForecast(forecast=0)               # illegal forecast value

    # test _step property working as expected
    seg = transform.SegmentXYForecast(width=10, overlap=0.5)
    assert seg._step == 5

    # test precedence of step over overlap
    seg = transform.SegmentXYForecast(width=10, overlap=1, step=1)
    assert seg._step == 1

    # illegal overlap value, but valid step value
    seg = transform.SegmentXYForecast(overlap=-1, step=1)
    assert seg._step == 1

    # test shape of segmented data
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
    # univariate time series
    N = 100
    t = np.arange(N) + np.random.rand(N)
    X = [np.column_stack([t, np.random.rand(N)])]
    y = [np.random.rand(N)]

    interp = transform.Interp(2)
    interp.fit(X)
    Xc, yc, swt = interp.transform(X, y)

    assert len(Xc[0]) == N / 2
    assert len(yc[0]) == N / 2
    assert np.ndim(Xc[0]) == 1

    y = [np.random.randint(0, 5, N)]
    interp = transform.Interp(5, kind='cubic', categorical_target=True)
    interp.fit(X, y)
    Xc, yc, swt = interp.transform(X, y)

    assert len(Xc[0]) == N / 5
    assert len(yc[0]) == N / 5
    assert np.ndim(Xc[0]) == 1
    assert np.all(np.isin(yc, np.arange(6)))

    # multivariate time series
    N = 100
    D = 5
    t = np.arange(N) + np.random.rand(N)
    X = [np.column_stack([t, np.random.rand(N,D)])]
    y = [np.random.rand(N)]

    interp = transform.Interp(2)
    interp.fit(X)
    Xc, yc, swt = interp.transform(X, y)

    assert len(Xc[0]) == N / 2
    assert len(yc[0]) == N / 2
    assert Xc[0].shape[1] == D

    y = [np.random.randint(0, 5, N)]
    interp = transform.Interp(5, kind='cubic', categorical_target=True)
    interp.fit(X, y)
    Xc, yc, swt = interp.transform(X, y)

    assert len(Xc[0]) == N / 5
    assert len(yc[0]) == N / 5
    assert Xc[0].shape[1] == D
    assert np.all(np.isin(yc, np.arange(6)))

    # sorting case
    N = 100
    t = np.arange(N)
    t[0:3] = 0
    X = [np.column_stack([t, np.random.rand(N)])]
    y = [np.random.rand(N)]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        interp = transform.Interp(sample_period=2, assume_sorted=False)
        interp.fit(X)
        Xc, yc, swt = interp.transform(X, y)
        assert len(w) == 2
        assert issubclass(w[-1].category, UserWarning)
        assert "duplicate" in str(w[-1].message)
        assert len(Xc[0]) == N / 2
        assert len(yc[0]) == N / 2
        assert np.ndim(Xc[0]) == 1
        assert np.count_nonzero(np.isnan(Xc)) == 0


def test_interp_long_to_wide():
    # Test 1
    t = np.array([1.1, 1.2, 2.1, 3.3, 3.4, 3.5]).astype(float)
    s = np.array([0, 1, 0, 0, 1, 1]).astype(float)
    v1 = np.array([3, 4, 5, 7, 15, 25]).astype(float)
    v2 = np.array([5, 7, 6, 9, 22, 35]).astype(float)
    y = np.array([1, 2, 2, 2, 3, 3]).astype(float)
    df = np.column_stack([t, s, v1, v2])

    X = [df, df]
    y = [y, y]
    
    stacked_interp = transform.InterpLongToWide(0.5)
    stacked_interp.fit(X, y)
    Xc, yc, swt = stacked_interp.transform(X, y)

    # --Checks--
    # linearly sampled time within bounds = 1.2, 1.7, 2.2, 2.7, 3.2 --> len(Xc[0]) = 5
    assert len(Xc[0]) == 5
    # Xc shape[1] = unique(s) * no. dimensions of values (V1) = 2 * 2 = 4
    assert Xc[0].shape[1] == 4
    assert swt is None

    # Test 2
    y = [1, 2]
    stacked_interp.fit(X, y)
    Xc, yc, swt = stacked_interp.transform(X, y)
    assert np.array_equal(yc, y)

    # Test 3
    N = 100
    sample_period = 0.5
    t = np.arange(N) + np.random.rand(N)
    s = np.array([1, 2] * int(N/2))
    np.random.shuffle(s)

    v1 = np.arange(N) + np.random.rand(N)
    v2 = np.arange(N) + np.random.rand(N)
    v3 = np.arange(N) + np.random.rand(N)
    df = np.column_stack([t, s, v1, v2, v3])
    X = [df, df, df]
    dm = np.arange(N) + np.random.rand(N)
    y = [dm, dm, dm]

    stacked_interp = transform.InterpLongToWide(sample_period)
    stacked_interp.fit(X, y)

    Xc, yc, swt = stacked_interp.transform(X, y)

    # --Checks--
    assert Xc[0].shape[1] == len(np.unique(s)) * (X[0].shape[1]-2)
    assert len(Xc[0]) <= N/sample_period

    # Test 3 - duplicate entries for t
    t = np.array([1.1, 1.1, 1.2, 2.1, 3.3, 3.4, 3.5]).astype(float)
    s = np.array([0, 0, 1, 0, 0, 1, 1]).astype(float)
    v1 = np.array([3, 3, 4, 5, 7, 15, 25]).astype(float)
    v2 = np.array([5, 5, 7, 6, 9, 22, 35]).astype(float)
    y = np.array([1, 1, 2, 2, 2, 3, 3]).astype(float)
    df = np.column_stack([t, s, v1, v2])

    X = [df, df]
    y = [y, y]

    with warnings.catch_warnings(record=True) as w:
        stacked_interp = transform.InterpLongToWide(0.5, assume_sorted=False)
        stacked_interp.fit(X, y)
        Xc, yc, swt = stacked_interp.transform(X, y)

        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "duplicate" in str(w[-1].message)

        # --Checks--
        assert len(Xc[0]) == 5
        assert Xc[0].shape[1] == 4
        assert swt is None
        assert np.count_nonzero(np.isnan(Xc)) == 0


def test_feature_rep_mix():
    union = transform.FeatureRepMix([
        ('a', transform.FeatureRep(features={'mean': mean}), 0),
        ('b', transform.FeatureRep(features={'mean': mean}), 1),
        ('c', transform.FeatureRep(features={'mean': mean}), [2,3]),
        ('d', transform.FeatureRep(features={'mean': mean}), slice(0,2)),
        ('e', transform.FeatureRep(features={'mean': mean}), [False, False, True, True]),
    ])

    # multivariate ts
    X = np.random.rand(100, 10, 4)
    y = np.ones(100)
    union.fit(X, y)
    Xt = union.transform(X)
    assert Xt.shape[0] == len(X)
    assert len(union.f_labels) == Xt.shape[1]

    # ts with multivariate contextual data
    X = TS_Data(np.random.rand(100, 10, 4), np.random.rand(100, 3))
    y = np.ones(100)
    union.fit(X, y)
    Xt = union.transform(X)
    assert Xt.shape[0] == len(X)
    assert len(union.f_labels) == Xt.shape[1]

    # ts with univariate contextual data
    X = TS_Data(np.random.rand(100, 10, 4), np.random.rand(100))
    y = np.ones(100)
    union.fit(X, y)
    Xt = union.transform(X)
    assert Xt.shape[0] == len(X)
    assert len(union.f_labels) == Xt.shape[1]

    # univariate ts
    uni_union = transform.FeatureRepMix([
        ('a', transform.FeatureRep(features={'mean': mean}), 0),
        ('b', transform.FeatureRep(features={'mean': mean}), [0]),
        ('c', transform.FeatureRep(features={'mean': mean}), slice(0,1)),
        ('d', transform.FeatureRep(features={'mean': mean}), [True]),
    ])
    X = np.random.rand(100, 10)
    y = np.ones(100)
    uni_union.fit(X, y)
    Xt = uni_union.transform(X)
    assert Xt.shape[0] == len(X)
    assert len(uni_union.f_labels) == Xt.shape[1]


def test_function_transform():
    constant = 10
    identity = transform.FunctionTransformer()
    def replace(Xt, value):
        return np.ones(Xt.shape) * value
    custom = transform.FunctionTransformer(replace, func_kwargs={"value": constant})

    # univariate ts
    X = np.random.rand(100, 10)
    y = np.ones(100)

    identity.fit(X, y)
    Xtrans = identity.transform(X)
    assert Xtrans is X

    custom.fit(X, y)
    Xtrans = custom.transform(X)
    assert np.array_equal(Xtrans, np.ones(X.shape) * constant)

    # multivariate ts
    X = np.random.rand(100, 10, 4)
    y = np.ones(100)

    identity.fit(X, y)
    Xtrans = identity.transform(X)
    assert Xtrans is X

    custom.fit(X, y)
    Xtrans = custom.transform(X)
    assert np.array_equal(Xtrans, np.ones(X.shape) * constant)

    # ts with univariate contextual data
    Xt = np.random.rand(100, 10, 4)
    Xc = np.random.rand(100)
    X = TS_Data(Xt, Xc)
    y = np.ones(100)

    identity.fit(X, y)
    Xtrans = identity.transform(X)
    assert Xtrans is X

    custom.fit(X, y)
    Xtrans = custom.transform(X)
    Xtt, Xtc = get_ts_data_parts(Xtrans)
    assert np.array_equal(Xtt, np.ones(Xt.shape) * constant)
    assert Xtc is Xc

    # ts with multivariate contextual data
    Xt = np.random.rand(100, 10, 4)
    Xc = np.random.rand(100, 3)
    X = TS_Data(Xt, Xc)
    y = np.ones(100)

    identity.fit(X, y)
    Xtrans = identity.transform(X)
    assert Xtrans is X

    custom.fit(X, y)
    Xtrans = custom.transform(X)
    Xtt, Xtc = get_ts_data_parts(Xtrans)
    assert np.array_equal(Xtt, np.ones(Xt.shape) * constant)
    assert Xtc is Xc

    # test resampling
    def resample(Xt):
        return Xt.reshape(1, -1)

    illegal_resampler = transform.FunctionTransformer(resample)
    X = np.random.rand(100, 10)
    y = np.ones(100)
    illegal_resampler.fit(X, y)
    with pytest.raises(ValueError):
        Xtrans = illegal_resampler.transform(X)
