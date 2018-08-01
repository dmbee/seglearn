'''
===========================
Resampling Time Series Data
===========================

This is a basic example using the pipeline to learn resample a time series

This may be useful for resampling irregularly sampled time series, or for determining
an optimal sampling frequency for the data

'''
# Author: David Burns
# License: BSD

from seglearn.datasets import load_watch
from seglearn.transform import FeatureRep, SegmentX, Interp
from seglearn.pipe import Pype

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, validation_curve

import matplotlib.pyplot as plt

import numpy as np

# seed RNGESUS
np.random.seed(123124)

# load the data
data = load_watch()

X = data['X']
y = data['y']

# I am adding in a column to represent time (50 Hz sampling), since my data doesn't include it
# the Interp class assumes time is the first column in the series
X = np.array([np.column_stack([np.arange(len(X[i]))/50.,X[i]]) for i in np.arange(len(X))])

clf = Pype([('interp', Interp(1./25., categorical_target=True)),
            ('segment', SegmentX()),
            ('features', FeatureRep()),
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier())])

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

clf.fit(X_train,y_train)
score = clf.score(X_test, y_test)

print("N series in train: ", len(X_train))
print("N series in test: ", len(X_test))
print("N segments in train: ", clf.N_train)
print("N segments in test: ", clf.N_test)
print("Accuracy score: ", score)

# lets try a few different sampling periods
periods = [1./5., 1./10., 1./25., 1./50.]
train_scores, test_scores = validation_curve(clf, X, y, param_name='interp__sample_period', param_range=periods, cv = 3)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Sample Period Validation Curve")
plt.xlabel("Sample Period [s]")
plt.ylabel("Accuracy")
plt.ylim(0.0, 1.1)

plt.plot(periods, train_scores_mean, label="Training score",color="darkorange", lw=2)
plt.fill_between(periods, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=2)
plt.plot(periods, test_scores_mean, label="Test score",color="navy", lw=2)
plt.fill_between(periods, test_scores_mean - train_scores_std,
                 test_scores_mean + train_scores_std, alpha=0.2,
                 color="navy", lw=2)
plt.legend(loc="best")
plt.show()