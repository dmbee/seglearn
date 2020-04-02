"""
===============================
Simple imbalanced-learn example
===============================

This example demonstrates how to use imbalanced-learn resample transforms inside a seglearn Pype.
"""

# Author: Matthias Gazzari
# License: BSD

import numpy as np

from sklearn.dummy import DummyClassifier

from seglearn.pipe import Pype
from seglearn.transform import Segment, patch_sampler, FeatureRep
from seglearn.feature_functions import minimum
from seglearn.split import temporal_split


from imblearn.under_sampling import RandomUnderSampler

# Single univariate time series with 10 samples
X = [np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5,6], [6, 7], [7, 8], [8, 9], [9, 10]])]
# Time series target (imbalanced towards False)
y = [np.array([True, False, False, False, False, False, True, False, False, False])]

print("Implementation details: transform and fit_transform methods:")

pipe = Pype([
    ('segment', Segment(width=1, overlap=0)),
    ('resample', patch_sampler(RandomUnderSampler)()),
])
print("Pipeline:", pipe)

print("Calling a transform on the data does not change it ...")
Xf, yf = pipe.transform(X, y)
print("X (flattened):", Xf.flatten())
print("y", yf)

print("... but calling fit_transform resamples the data.")
Xf, yf = pipe.fit_transform(X, y)
print("X (flattened):", Xf.flatten())
print("y", yf)

print()
print("VerboseDummyClassifier example:")
print()

class VerboseDummyClassifier(DummyClassifier):
    def fit(self, X, y, sample_weight=None):
        print("Fitting X (flattened):", X.flatten(), "on y:", y)
        return super(VerboseDummyClassifier, self).fit(X, y, sample_weight)
    def predict(self, X):
        print("Predicting X (flattened):", X.flatten())
        return super(VerboseDummyClassifier, self).predict(X)
    def score(self, X, y, sample_weight=None):
        print("Scoring X (flattened):", X.flatten(), "on y:", y)
        return super(VerboseDummyClassifier, self).score(X, y, sample_weight)

pipe = Pype([
    ('segment', Segment(width=1, overlap=0)),
    ('resample', patch_sampler(RandomUnderSampler)(shuffle=True)),
    ('feature', FeatureRep(features={"min":minimum})),
    ('estimator', VerboseDummyClassifier(strategy="constant", constant=True)),
])
print("Pipeline:", pipe)

print("Split the data into half training and half test data:")
X_train, X_test, y_train, y_test = temporal_split(X, y, 0.5)
print("X_train:", X_train)
print("y_train:", y_train)
print("X_test:", X_test)
print("y_test:", y_test)
print()

print("Fit on the training data (this includes resampling):")
pipe.fit(X_train, y_train)
print()

print("Score the fitted estimator on test data (this excludes resampling):")
score = pipe.score(X_test, y_test)
print("Score: ", score)
