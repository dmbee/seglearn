from seglearn.base import TS_Data
import numpy as np

def test_ts_data():
    # time series data
    ts = np.array([np.random.rand(100,10),np.random.rand(200,10),np.random.rand(20,10)])
    c = np.random.rand(3,10)

    data = TS_Data(ts, c)

    assert type(data[1]) == TS_Data

    # segmented time series data

    sts = np.random.rand(100,10,6)
    c = np.random.rand(100, 6)

    data = TS_Data(sts, c)
    assert type(data[4:10]) == TS_Data


    sts = np.random.rand(100,10)
    c = np.random.rand(100)

    data = TS_Data(sts, c)
    assert type(data[4:10]) == TS_Data