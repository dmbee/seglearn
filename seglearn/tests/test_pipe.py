from seglearn.pipe import SegPipe
from seglearn.transform import FeatureRep, SegmentX, SegmentXY
from seglearn.util import make_ts_data

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
import numpy as np


def test_pipe_classification():
    # no context data, single time series
    X = [np.random.rand(1000,10)]
    y = [5]
    est = Pipeline([('ftr', FeatureRep()),
                    ('ridge', RandomForestClassifier())])

    pipe = SegPipe(est, segmenter=SegmentX())

    pipe.fit(X, y)
    pipe.predict(X, y)
    pipe.score(X, y)

    # context data, single time seres
    Xt = [np.random.rand(1000, 10)]
    Xc = [np.random.rand(3)]
    X = make_ts_data(Xt, Xc)
    y = [5]

    pipe.fit(X, y)
    pipe.predict(X, y)
    pipe.score(X, y)

    # multiple time series
    Xt = [np.random.rand(1000, 10), np.random.rand(100, 10), np.random.rand(500, 10)]
    Xc = np.random.rand(3,3)
    X = make_ts_data(Xt, Xc)
    y = [1,2,3]

    pipe.fit(X, y)
    pipe.predict(X, y)
    pipe.score(X, y)

    # univariate data
    Xt = [np.random.rand(1000), np.random.rand(100), np.random.rand(500)]
    Xc = np.random.rand(1,3)
    y = [1,2,3]

    pipe.fit(X, y)
    pipe.predict(X, y)
    pipe.score(X, y)




def test_pipe_regression():
    # no context data, single time series
    X = [np.random.rand(1000,10)]
    y = [np.random.rand(1000)]
    est = Pipeline([('ftr', FeatureRep()),
                    ('ridge', Ridge())])

    pipe = SegPipe(est, segmenter=SegmentXY())

    pipe.fit(X, y)
    pipe.predict(X, y)
    pipe.score(X, y)


    # context data, single time seres
    Xt = [np.random.rand(1000, 10)]
    Xc = [np.random.rand(3)]
    X = make_ts_data(Xt, Xc)
    y = [np.random.rand(1000)]

    pipe.fit(X, y)
    pipe.predict(X, y)
    pipe.score(X, y)

    # multiple time seres
    Xt = [np.random.rand(1000, 10), np.random.rand(100, 10), np.random.rand(500, 10)]
    Xc = np.random.rand(3,3)
    X = make_ts_data(Xt, Xc)
    y = [np.random.rand(1000), np.random.rand(100), np.random.rand(500)]

    pipe.fit(X, y)
    pipe.predict(X, y)
    pipe.score(X, y)

    # cross val

    Xt = np.array([np.random.rand(1000,10) for i in range(5)])
    Xc = np.random.rand(5,3)
    X = make_ts_data(Xt, Xc)
    y = np.array([np.random.rand(1000) for i in range(5)])

    cross_validate(pipe, X, y)



