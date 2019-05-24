# Author: David Burns
# License: BSD

import numpy as np
import pandas as pd

from seglearn.preprocessing import TargetRunLengthEncoder
from seglearn.base import TS_Data

from seglearn.util import get_ts_data_parts

def test_trle():

    # Multivariate data
    Nt = 100
    nvars = 5
    X = [np.random.rand(Nt, nvars)]
    y = [np.concatenate([np.full(3, 1), np.full(26, 2), np.full(1, 3), np.full(70, 4)])]

    rle = TargetRunLengthEncoder(min_length=5)
    rle.fit(X)
    Xt, yt, _ = rle.transform(X, y)

    assert len(Xt) == len(yt) and len(yt) == 2
    assert yt[0] == 2 and yt[1] == 4
    assert len(Xt[0]) == 26 and len(Xt[1]) == 70

    # Nothing excluded
    Nt = 100
    nvars = 5
    X = [np.random.rand(Nt, nvars)]
    y = [np.concatenate([np.full(50,1), np.full(50,2)])]

    rle = TargetRunLengthEncoder(min_length=5)
    rle.fit(X, y)
    Xt, yt, _ = rle.transform(X, y)

    assert len(Xt) == len(yt) and len(yt) == 2
    assert np.all(np.concatenate(Xt) == X)
    assert yt[0] == 1 and yt[1] == 2
    assert len(Xt[0]) == 50 and len(Xt[1]) == 50

    # Univariate data with sample weight and context
    Nt = 100
    Xts = [np.random.rand(Nt)]
    Xc = [5]
    X = TS_Data(Xts, Xc)
    y = [np.concatenate([np.full(3, 1), np.full(26, 2), np.full(1, 3), np.full(70, 4)])]
    sw = [1]

    rle = TargetRunLengthEncoder(min_length=5)
    rle.fit(X)
    Xt, yt, swt = rle.transform(X, y, sw)
    Xtc = Xt.context_data

    assert len(Xt) == len(yt) and len(swt) == len(yt) and len(yt) == 2
    assert yt[0] == 2 and yt[1] == 4
    assert len(Xt[0]) == 26 and len(Xt[1]) == 70
    assert swt[0] == 1 and swt[1] == 1
    assert Xtc[0] == 5 and Xtc[1] == 5

    X = pd.DataFrame(Xc)
    X['ts_data'] = Xts
    X = TS_Data.from_df(X)

    rle = TargetRunLengthEncoder(min_length=5)
    rle.fit(X)
    Xt, yt, swt = rle.transform(X, y, sw)
    Xtc = Xt.context_data

    assert len(Xt) == len(yt) and len(swt) == len(yt) and len(yt) == 2
    assert yt[0] == 2 and yt[1] == 4
    assert len(Xt[0]) == 26 and len(Xt[1]) == 70
    assert swt[0] == 1 and swt[1] == 1
    assert Xtc[0] == 5 and Xtc[1] == 5






