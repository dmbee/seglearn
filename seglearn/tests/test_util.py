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
