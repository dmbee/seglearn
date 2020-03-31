# Author: David Burns
# License: BSD

import numpy as np

from seglearn.datasets import load_watch
from seglearn.base import TS_Data
from seglearn import util


def test_util():
    df = load_watch()

    data = TS_Data(df['X'], df['side'])
    Xt, Xc = util.get_ts_data_parts(data)

    assert np.array_equal(Xc, df['side'])
    assert np.all([np.array_equal(Xt[i], df['X'][i]) for i in range(len(df['X']))])

    util.check_ts_data(data, df['y'])
    util.check_ts_data(df['X'], df['y'])

    util.ts_stats(df['X'], df['y'], fs=1., class_labels=df['y_labels'])


def test_to_categorical_series():
    p = np.arange(10)
    step = 2
    width = 3
    s = util.segmented_prediction_to_series(p, step, width, categorical_target=True)
    assert len(s) == (len(p) - 1) * step + width
    assert np.all(np.isin(s, p))

    p = np.arange(10)
    step = 3
    width = 2
    s = util.segmented_prediction_to_series(p, step, width, categorical_target=True)
    assert len(s) == (len(p) - 1) * step + width
    assert np.all(np.isin(s, p))

    p = np.arange(10)
    p = np.column_stack([p, p])
    step = 2
    width = 3
    s = util.segmented_prediction_to_series(p, step, width, categorical_target=True)
    assert len(s) == (len(p) - 1) * step + width
    assert s.shape[1] == 2
    assert np.all(np.isin(s, p))

def test_to_real_series():
    p = np.arange(20)  # highly overlapping case
    step = 2
    width = 5
    s = util.segmented_prediction_to_series(p, step, width, categorical_target=False)
    assert len(s) == (len(p) - 1) * step + width
    assert np.all(s <= max(p))
    assert np.all(s >= min(p))

    p = np.arange(10)
    step = 3
    width = 2
    s = util.segmented_prediction_to_series(p, step, width, categorical_target=False)
    assert len(s) == (len(p) - 1) * step + width
    assert np.all(s <= max(p))
    assert np.all(s >= min(p))

    p = np.arange(5)
    p = np.column_stack([p, p])
    step = 2
    width = 5
    s = util.segmented_prediction_to_series(p, step, width, categorical_target=False)
    assert len(s) == (len(p) - 1) * step + width
    assert s.shape[1] == 2
    assert np.all(s <= np.max(p))
    assert np.all(s >= np.min(p))

