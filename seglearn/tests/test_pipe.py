# Author: David Burns
# License: BSD

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler

from seglearn.base import TS_Data
from seglearn.pipe import Pype
from seglearn.transform import FeatureRep, SegmentX, SegmentXY, SegmentXYForecast, PadTrunc


def yvals(y):
    if len(np.atleast_1d(y[0])) > 1:
        return np.unique(np.concatenate(y))
    else:
        return np.unique(y)


def transformation_test(clf, X, y):
    clf.fit(X, y)
    Xtr1, ytr1 = clf.transform(X, y)
    Xtr2, ytr2 = clf.fit_transform(X, y)
    assert np.all(Xtr1 == Xtr2)
    assert np.all(ytr1 == ytr2)
    assert np.all(np.isin(np.unique(ytr1), yvals(y)))
    assert len(Xtr1) == len(ytr1)


@pytest.mark.filterwarnings("ignore:deprecated, use Segment class")
def test_pipe_transformation():
    # SegmentX transform pipe
    pipe = Pype([('seg', SegmentX()),
                 ('ftr', FeatureRep()),
                 ('scaler', StandardScaler())])
    Xt = [np.random.rand(1000, 10), np.random.rand(100, 10), np.random.rand(500, 10)]
    Xc = np.random.rand(3, 3)
    X = TS_Data(Xt, Xc)
    y = [1, 2, 3]
    transformation_test(pipe, X, y)

    X = pd.DataFrame(Xc)
    X['ts_data'] = Xt
    X = TS_Data.from_df(X)
    transformation_test(pipe, X, y)

    # SegmentXY transform pipe
    pipe = Pype([('seg', SegmentXY()),
                 ('ftr', FeatureRep()),
                 ('scaler', StandardScaler())])
    Xt = [np.random.rand(1000, 10), np.random.rand(100, 10), np.random.rand(500, 10)]
    Xc = np.random.rand(3, 3)
    X = TS_Data(Xt, Xc)
    y = [np.random.rand(1000), np.random.rand(100), np.random.rand(500)]
    transformation_test(pipe, X, y)

    X = pd.DataFrame(Xc)
    X['ts_data'] = Xt
    X = TS_Data.from_df(X)
    transformation_test(pipe, X, y)

    # Forecast transform pipe
    pipe = Pype([('seg', SegmentXYForecast()),
                 ('ftr', FeatureRep()),
                 ('scaler', StandardScaler())])
    Xt = [np.random.rand(1000, 10), np.random.rand(100, 10), np.random.rand(500, 10)]
    Xc = np.random.rand(3, 3)
    X = TS_Data(Xt, Xc)
    y = [np.random.rand(1000), np.random.rand(100), np.random.rand(500)]
    transformation_test(pipe, X, y)

    X = pd.DataFrame(Xc)
    X['ts_data'] = Xt
    X = TS_Data.from_df(X)
    transformation_test(pipe, X, y)

    # Padtrunc transform pipe
    pipe = Pype([('trunc', PadTrunc()),
                 ('ftr', FeatureRep()),
                 ('scaler', StandardScaler())])
    Xt = [np.random.rand(1000, 10), np.random.rand(100, 10), np.random.rand(500, 10)]
    Xc = np.random.rand(3, 3)
    X = TS_Data(Xt, Xc)
    y = [1, 2, 3]
    transformation_test(pipe, X, y)

    X = pd.DataFrame(Xc)
    X['ts_data'] = Xt
    X = TS_Data.from_df(X)
    transformation_test(pipe, X, y)


def classifier_test(clf, X, y):
    yv = yvals(y)
    clf.fit(X, y)
    yp = clf.predict(X)
    ytr, yp2 = clf.transform_predict(X, y)
    assert np.all(np.isin(np.unique(ytr), yv))
    assert len(ytr) == len(yp2)
    assert np.all(np.isin(np.unique(yp2), yv))
    assert np.all(yp == yp2)
    pp = clf.predict_proba(X)
    assert pp.shape[0] == len(yp)
    assert pp.shape[1] == len(yv)
    score = clf.score(X, y)
    assert 1.0 >= score >= 0.0

    ys = clf.predict_as_series(X)
    assert len(X) == len(ys)

    if clf._get_segmenter():
        s = clf.predict_unsegmented(X, categorical_target=True)
        for i in np.arange(len(X)):
            assert len(X[i]) == len(s[i])
            assert np.all(np.isin(np.unique(s[i]), yv))


@pytest.mark.filterwarnings("ignore:deprecated, use Segment class")
def test_pipe_classification():
    # no context data, single time series
    X = [np.random.rand(1000, 10)]
    y = [5]

    pipe = Pype([('seg', SegmentX()),
                 ('ftr', FeatureRep()),
                 ('rf', RandomForestClassifier(n_estimators=10))])

    classifier_test(pipe, X, y)

    # context data, single time seres
    Xt = [np.random.rand(1000, 10)]
    Xc = [np.random.rand(3)]
    X = TS_Data(Xt, Xc)
    y = [5]
    classifier_test(pipe, X, y)

    X = pd.DataFrame(Xc)
    X['ts_data'] = Xt
    X = TS_Data.from_df(X)
    classifier_test(pipe, X, y)

    # multiple time series
    Xt = [np.random.rand(1000, 10), np.random.rand(100, 10), np.random.rand(500, 10)]
    Xc = np.random.rand(3, 3)
    X = TS_Data(Xt, Xc)
    y = [1, 2, 3]
    classifier_test(pipe, X, y)

    X = pd.DataFrame(Xc)
    X['ts_data'] = Xt
    X = TS_Data.from_df(X)
    classifier_test(pipe, X, y)

    # univariate data
    Xt = [np.random.rand(1000), np.random.rand(100), np.random.rand(500)]
    Xc = np.random.rand(3)
    X = TS_Data(Xt, Xc)
    y = [1, 2, 3]
    classifier_test(pipe, X, y)

    X = pd.DataFrame(Xc)
    X['ts_data'] = Xt
    X = TS_Data.from_df(X)
    classifier_test(pipe, X, y)


def regression_test(clf, X, y):
    yv = yvals(y)
    clf.fit(X, y)
    yp = clf.predict(X)
    ytr, yp2 = clf.transform_predict(X, y)
    assert np.all(np.isin(np.unique(ytr), yv))
    assert len(ytr) == len(yp2)
    assert np.all(yp == yp2)
    score = clf.score(X, y)
    assert 1.0 >= score >= 0.0

    ys = clf.predict_as_series(X)
    assert len(X) == len(ys)

    if clf._get_segmenter():
        s = clf.predict_unsegmented(X, categorical_target=False)
        for i in np.arange(len(X)):
            assert len(X[i]) == len(s[i])
            assert np.max(yp) >= np.max(s[i])
            assert np.min(yp) <= np.min(s[i])


@pytest.mark.filterwarnings("ignore:deprecated, use Segment class")
def test_pipe_regression():
    # no context data, single time series
    X = [np.random.rand(1000, 10)]
    y = [np.random.rand(1000)]
    pipe = Pype([('seg', SegmentXY()),
                 ('ftr', FeatureRep()),
                 ('ridge', Ridge())])
    regression_test(pipe, X, y)

    # context data, single time seres
    Xt = [np.random.rand(1000, 10)]
    Xc = [np.random.rand(3)]
    X = TS_Data(Xt, Xc)
    y = [np.random.rand(1000)]
    regression_test(pipe, X, y)

    X = pd.DataFrame(Xc)
    X['ts_data'] = Xt
    X = TS_Data.from_df(X)
    regression_test(pipe, X, y)

    # multiple time seres
    Xt = [np.random.rand(1000, 10), np.random.rand(100, 10), np.random.rand(500, 10)]
    Xc = np.random.rand(3, 3)
    X = TS_Data(Xt, Xc)
    y = [np.random.rand(1000), np.random.rand(100), np.random.rand(500)]
    regression_test(pipe, X, y)

    X = pd.DataFrame(Xc)
    X['ts_data'] = Xt
    X = TS_Data.from_df(X)
    regression_test(pipe, X, y)

    # cross val
    Xt = np.array([np.random.rand(1000, 10)] * 5)
    Xc = np.random.rand(5, 3)
    X = TS_Data(Xt, Xc)
    y = np.array([np.random.rand(1000)] * 5)
    cross_validate(pipe, X, y, cv=3)

    X = pd.DataFrame(Xc)
    Xt = [np.random.rand(1000, 10)] * 5
    X['ts_data'] = Xt
    X = TS_Data.from_df(X)

    cross_validate(pipe, X, y, cv=3)


def forecast_test(clf, X, y):
    yv = yvals(y)
    clf.fit(X, y)
    yp = clf.predict(X)
    ytr, yp2 = clf.transform_predict(X, y)
    assert np.all(np.isin(np.unique(ytr), yv))
    assert len(ytr) == len(yp2)
    assert np.all(yp == yp2)
    score = clf.score(X, y)
    assert 1.0 >= score >= 0.0


def test_pipe_forecast():
    # no context data, single time series
    X = [np.random.rand(1000, 10)]
    y = [np.random.rand(1000)]

    pipe = Pype([('seg', SegmentXYForecast()),
                 ('ftr', FeatureRep()),
                 ('ridge', Ridge())])

    forecast_test(pipe, X, y)

    # context data, single time seres
    Xt = [np.random.rand(1000, 10)]
    Xc = [np.random.rand(3)]
    X = TS_Data(Xt, Xc)
    y = [np.random.rand(1000)]

    forecast_test(pipe, X, y)

    X = pd.DataFrame(Xc)
    X['ts_data'] = Xt
    X = TS_Data.from_df(X)
    forecast_test(pipe, X, y)

    # multiple time seres
    Xt = [np.random.rand(1000, 10), np.random.rand(100, 10), np.random.rand(500, 10)]
    Xc = np.random.rand(3, 3)
    X = TS_Data(Xt, Xc)
    y = [np.random.rand(1000), np.random.rand(100), np.random.rand(500)]

    forecast_test(pipe, X, y)

    X = pd.DataFrame(Xc)
    X['ts_data'] = Xt
    X = TS_Data.from_df(X)
    forecast_test(pipe, X, y)

    # cross val

    Xt = np.array([np.random.rand(1000, 10)] * 5)
    Xc = np.random.rand(5, 3)
    X = TS_Data(Xt, Xc)
    y = np.array([np.random.rand(1000)] * 5)

    cross_validate(pipe, X, y, cv=3)

    X = pd.DataFrame(Xc)
    Xt = [np.random.rand(1000, 10)] * 5
    X['ts_data'] = Xt
    X = TS_Data.from_df(X)
    cross_validate(pipe, X, y, cv=3)


def test_pipe_PadTrunc():
    # no context data, single time series
    X = [np.random.rand(1000, 10)]
    y = [5]
    pipe = Pype([('trunc', PadTrunc()),
                 ('ftr', FeatureRep()),
                 ('rf', RandomForestClassifier(n_estimators=10))])
    classifier_test(pipe, X, y)

    # context data, single time seres
    Xt = [np.random.rand(1000, 10)]
    Xc = [np.random.rand(3)]
    X = TS_Data(Xt, Xc)
    y = [5]
    classifier_test(pipe, X, y)

    X = pd.DataFrame(Xc)
    X['ts_data'] = Xt
    classifier_test(pipe, X, y)

    # multiple time series
    Xt = [np.random.rand(1000, 10), np.random.rand(100, 10), np.random.rand(500, 10)]
    Xc = np.random.rand(3, 3)
    X = TS_Data(Xt, Xc)
    y = [1, 2, 3]
    classifier_test(pipe, X, y)

    X = pd.DataFrame(Xc)
    X['ts_data'] = Xt
    classifier_test(pipe, X, y)

    # univariate data
    Xt = [np.random.rand(1000), np.random.rand(100), np.random.rand(500)]
    Xc = np.random.rand(3)
    X = TS_Data(Xt, Xc)
    y = [1, 2, 3]
    classifier_test(pipe, X, y)

    X = pd.DataFrame(Xc)
    X['ts_data'] = Xt
    X = TS_Data.from_df(X)
    classifier_test(pipe, X, y)

