# Author: David Burns
# License: BSD

import numpy as np

from seglearn.datasets import load_watch
from seglearn.base import TS_Data


def test_ts_data():
    # time series data
    ts = np.array([np.random.rand(100, 10), np.random.rand(200, 10), np.random.rand(20, 10)])
    c = np.random.rand(3, 10)
    data = TS_Data(ts, c)

    assert np.array_equal(data.context_data, c)
    assert np.array_equal(data.ts_data, ts)

    assert isinstance(data[1], TS_Data)
    assert np.array_equal(data[1].ts_data, ts[1])
    assert np.array_equal(data[1].context_data, c[1])

    # segmented time series data

    sts = np.random.rand(100, 10, 6)
    c = np.random.rand(100, 6)

    data = TS_Data(sts, c)
    assert isinstance(data[4:10], TS_Data)
    assert np.array_equal(data[4:10].ts_data, sts[4:10])
    assert np.array_equal(data[4:10].context_data, c[4:10])

    sts = np.random.rand(100, 10)
    c = np.random.rand(100)

    data = TS_Data(sts, c)
    assert isinstance(data[4:10], TS_Data)
    assert np.array_equal(data[4:10].ts_data, sts[4:10])
    assert np.array_equal(data[4:10].context_data, c[4:10])


def test_watch():
    df = load_watch()
    data = TS_Data(df['X'], df['side'])
    assert isinstance(data, TS_Data)
