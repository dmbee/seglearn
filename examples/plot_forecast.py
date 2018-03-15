'''
=======================
Time Series Forecasting
=======================

In this example, we use a feature representation pipeline to forecast a continuous time series
target with a regressor.

In this example, we train the algorithm from the features and targets in the training set.
Then predict (future segments) from the features in the test set.

We do not sequentially retrain the algorithm as we move through the test set - which is an
approach you will sometimes see with time series forecasting (and which may or may not be
useful in your application).

'''
# Author: David Burns
# License: BSD


from seglearn.transform import FeatureRep, SegmentXY, mean
from seglearn.pipe import SegPipe
from seglearn.split import temporal_split

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

import numpy as np

# remember for a single time series, we need to make it a list
X = [np.arange(5000)/100.]
y = [np.sin(X[0])*X[0]*2.5 + X[0]*X[0]]

# split the data along the time axis (our only option since we have only 1 time series)
X_train, X_test, y_train, y_test = temporal_split(X, y, test_size=0.25)

# create a feature representation pipeline
est = Pipeline([('features', FeatureRep()),
                ('lin', LinearRegression())])

# setting y_func = mean, selects the mean from each y segment as the target
# forecast = 8 with overlap 0.5 means we are predicting 4 segments ahead (800 time points)
# see the API documentation for further details
segmenter = SegmentXY(width = 200, overlap=0.5, y_func=mean, forecast=8)
pipe = SegPipe(est, segmenter)

# fit and score
pipe.fit(X_train,y_train)
score = pipe.score(X_test, y_test)

print("N series in train: ", len(X_train))
print("N series in test: ", len(X_test))
print("N segments in train: ", pipe.N_train)
print("N segments in test: ", pipe.N_test)
print("Score: ", score)

# generate some predictions
y, y_p = pipe.predict(X, y) # all predictions
ytr, ytr_p = pipe.predict(X_train, y_train) # training predictions
yte, yte_p = pipe.predict(X_test, y_test) # test predictions

# for demonstration, we'll use the complete prediction set
# to avoid a discontinuity in the plot
y = y[len(ytr):len(y)]
y_p = y_p[len(ytr):len(y_p)]
xtr = np.arange(len(ytr)) # segment number
x = np.arange(len(y)) + len(xtr)

# plot the results
plt.plot(xtr, ytr, '.', label = "training")
plt.plot(x, y, '.', label ="actual")
plt.plot(x, y_p, label ="predicted")
plt.xlabel("Segment Number")
plt.ylabel("Target")
plt.legend()
plt.show()

