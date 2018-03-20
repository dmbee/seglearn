# Author: David Burns
# License: BSD

from seglearn.datasets import load_watch
from seglearn import util

def test_util():
    df = load_watch()

    data = util.make_ts_data(df['X'], df['side'])
    util.get_ts_data_parts(data)

    util.check_ts_data(data, df['y'])
    util.check_ts_data(df['X'], df['y'])

    util.ts_stats(df['X'], df['y'], fs = 1., class_labels=df['y_labels'])






