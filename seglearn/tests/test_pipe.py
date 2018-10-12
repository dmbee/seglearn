# Author: David Burns
# License: BSD

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

from seglearn.pipe import Pype
from seglearn.transform import FeatureRep, SegmentX, SegmentXY, SegmentXYForecast, PadTrunc
from seglearn.base import TS_Data


def test_pipe_classification():
    # no context data, single time series
    X = [np.random.rand(1000, 10)]
    y = [5]

    pipe = Pype([('seg', SegmentX()),
                 ('ftr', FeatureRep()),
                 ('rf', RandomForestClassifier(n_estimators=10))])

    pipe.fit(X, y)
    pipe.predict(X)
    pipe.transform_predict(X, y)
    pipe.predict_proba(X)
    pipe.predict_log_proba(X)
    pipe.score(X, y)

    # context data, single time seres
    Xt = [np.random.rand(1000, 10)]
    Xc = [np.random.rand(3)]
    X = TS_Data(Xt, Xc)
    y = [5]

    pipe.fit(X, y)
    pipe.transform_predict(X, y)
    pipe.predict(X)
    pipe.score(X, y)

    # multiple time series
    Xt = [np.random.rand(1000, 10), np.random.rand(100, 10), np.random.rand(500, 10)]
    Xc = np.random.rand(3, 3)
    X = TS_Data(Xt, Xc)
    y = [1, 2, 3]

    pipe.fit(X, y)
    pipe.transform_predict(X, y)
    pipe.predict(X)
    pipe.score(X, y)

    # univariate data
    Xt = [np.random.rand(1000), np.random.rand(100), np.random.rand(500)]
    Xc = np.random.rand(3)
    X = TS_Data(Xt, Xc)
    y = [1, 2, 3]

    pipe.fit(X, y)
    pipe.transform_predict(X, y)
    pipe.predict(X)
    pipe.score(X, y)

    # transform pipe
    pipe = Pype([('seg', SegmentX()),
                 ('ftr', FeatureRep()),
                 ('scaler', StandardScaler())])

    Xt = [np.random.rand(1000, 10), np.random.rand(100, 10), np.random.rand(500, 10)]
    Xc = np.random.rand(3, 3)
    X = TS_Data(Xt, Xc)
    y = [1, 2, 3]

    pipe.fit(X, y)
    pipe.transform(X, y)
    pipe.fit_transform(X, y)


def test_pipe_regression():
    # no context data, single time series
    X = [np.random.rand(1000, 10)]
    y = [np.random.rand(1000)]

    pipe = Pype([('seg', SegmentXY()),
                 ('ftr', FeatureRep()),
                 ('ridge', Ridge())])

    pipe.fit(X, y)
    pipe.transform_predict(X, y)
    pipe.predict(X)
    pipe.score(X, y)

    # context data, single time seres
    Xt = [np.random.rand(1000, 10)]
    Xc = [np.random.rand(3)]
    X = TS_Data(Xt, Xc)
    y = [np.random.rand(1000)]

    pipe.fit(X, y)
    pipe.transform_predict(X, y)
    pipe.predict(X)
    pipe.score(X, y)

    # multiple time seres
    Xt = [np.random.rand(1000, 10), np.random.rand(100, 10), np.random.rand(500, 10)]
    Xc = np.random.rand(3, 3)
    X = TS_Data(Xt, Xc)
    y = [np.random.rand(1000), np.random.rand(100), np.random.rand(500)]

    pipe.fit(X, y)
    pipe.transform_predict(X, y)
    pipe.predict(X)
    pipe.score(X, y)

    # cross val
    Xt = np.array([np.random.rand(1000, 10)] * 5)
    Xc = np.random.rand(5, 3)
    X = TS_Data(Xt, Xc)
    y = np.array([np.random.rand(1000)] * 5)

    cross_validate(pipe, X, y, cv=3)

    # transform pipe
    pipe = Pype([('seg', SegmentXY()),
                 ('ftr', FeatureRep()),
                 ('scaler', StandardScaler())])

    Xt = [np.random.rand(1000, 10), np.random.rand(100, 10), np.random.rand(500, 10)]
    Xc = np.random.rand(3, 3)
    X = TS_Data(Xt, Xc)
    y = [np.random.rand(1000), np.random.rand(100), np.random.rand(500)]

    pipe.fit(X, y)
    pipe.transform(X, y)
    pipe.fit_transform(X, y)


def test_pipe_forecast():
    # no context data, single time series
    X = [np.random.rand(1000, 10)]
    y = [np.random.rand(1000)]

    pipe = Pype([('seg', SegmentXYForecast()),
                 ('ftr', FeatureRep()),
                 ('ridge', Ridge())])

    pipe.fit(X, y)
    pipe.transform_predict(X, y)
    pipe.predict(X)
    pipe.score(X, y)

    # context data, single time seres
    Xt = [np.random.rand(1000, 10)]
    Xc = [np.random.rand(3)]
    X = TS_Data(Xt, Xc)
    y = [np.random.rand(1000)]

    pipe.fit(X, y)
    pipe.transform_predict(X, y)
    pipe.predict(X)
    pipe.score(X, y)

    # multiple time seres
    Xt = [np.random.rand(1000, 10), np.random.rand(100, 10), np.random.rand(500, 10)]
    Xc = np.random.rand(3, 3)
    X = TS_Data(Xt, Xc)
    y = [np.random.rand(1000), np.random.rand(100), np.random.rand(500)]

    pipe.fit(X, y)
    pipe.transform_predict(X, y)
    pipe.predict(X)
    pipe.score(X, y)

    # cross val

    Xt = np.array([np.random.rand(1000, 10)] * 5)
    Xc = np.random.rand(5, 3)
    X = TS_Data(Xt, Xc)
    y = np.array([np.random.rand(1000)] * 5)

    cross_validate(pipe, X, y, cv=3)

    # transform pipe
    pipe = Pype([('seg', SegmentXYForecast()),
                 ('ftr', FeatureRep()),
                 ('scaler', StandardScaler())])

    Xt = [np.random.rand(1000, 10), np.random.rand(100, 10), np.random.rand(500, 10)]
    Xc = np.random.rand(3, 3)
    X = TS_Data(Xt, Xc)
    y = [np.random.rand(1000), np.random.rand(100), np.random.rand(500)]

    pipe.fit(X, y)
    pipe.transform(X, y)
    pipe.fit_transform(X, y)


def test_pipe_PadTrunc():
    # no context data, single time series
    X = [np.random.rand(1000, 10)]
    y = [5]

    pipe = Pype([('trunc', PadTrunc()),
                 ('ftr', FeatureRep()),
                 ('rf', RandomForestClassifier(n_estimators=10))])

    pipe.fit(X, y)
    pipe.transform_predict(X, y)
    pipe.predict(X)
    pipe.score(X, y)

    # context data, single time seres
    Xt = [np.random.rand(1000, 10)]
    Xc = [np.random.rand(3)]
    X = TS_Data(Xt, Xc)
    y = [5]

    pipe.fit(X, y)
    pipe.transform_predict(X, y)
    pipe.predict(X)
    pipe.score(X, y)

    # multiple time series
    Xt = [np.random.rand(1000, 10), np.random.rand(100, 10), np.random.rand(500, 10)]
    Xc = np.random.rand(3, 3)
    X = TS_Data(Xt, Xc)
    y = [1, 2, 3]

    pipe.fit(X, y)
    pipe.transform_predict(X, y)
    pipe.predict(X)
    pipe.score(X, y)

    # univariate data
    Xt = [np.random.rand(1000), np.random.rand(100), np.random.rand(500)]
    Xc = np.random.rand(3)
    X = TS_Data(Xt, Xc)
    y = [1, 2, 3]

    pipe.fit(X, y)
    pipe.transform_predict(X, y)
    pipe.predict(X)
    pipe.score(X, y)

    # transform pipe   
    pipe = Pype([('trunc', PadTrunc()),
                 ('ftr', FeatureRep()),
                 ('scaler', StandardScaler())])

    Xt = [np.random.rand(1000, 10), np.random.rand(100, 10), np.random.rand(500, 10)]
    Xc = np.random.rand(3, 3)
    X = TS_Data(Xt, Xc)
    y = [1, 2, 3]

    pipe.fit(X, y)
    pipe.transform(X, y)
    pipe.fit_transform(X, y)
