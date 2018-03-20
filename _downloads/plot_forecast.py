'''
=======================
Time Series Forecasting
=======================

In this example, we use a feature representation pipeline to forecast a continuous time series
target with a regressor.

The algorithm is trained from the target from the features and targets in the training set.
Then predict (future segments) from the features in the test set.

We do not sequentially retrain the algorithm as we move through the test set - which is an
approach you will sometimes see with time series forecasting (and which may or may not be
useful in your application).

'''
# Author: David Burns
# License: BSD


from seglearn.transform import FeatureRep, SegmentXYForecast, last
from seglearn.pipe import SegPipe
from seglearn.split import temporal_split

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

import numpy as np

t = np.arange(5000)/100.
y = np.sin(t)*t*2.5 + t*t

# with forecasting, X can include the target
X = np.stack([t, y], axis = 1)

# remember for a single time series, we need to make a list
X = [X]
y = [y]

# split the data along the time axis (our only option since we have only 1 time series)
X_train, X_test, y_train, y_test = temporal_split(X, y, test_size=0.25)

# create a feature representation pipeline
est = Pipeline([('features', FeatureRep()),
                ('lin', LinearRegression())])

# setting y_func = last, and forecast = 200 makes us predict the value of y
# 200 samples ahead of the segment
# other reasonable options for y_func are ``mean``, ``all`` (or create your own function)
# see the API documentation for further details
segmenter = SegmentXYForecast(width = 200, overlap=0.5, y_func=last, forecast=200)
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


# note - the first few segments in the test set won't have predictions (gap)
# we plot the 'gap' for the visualization to hopefully make the situation clear
Ns = len(y)
ts = np.arange(Ns) # segment number
ttr = ts[0:len(ytr)]
tte = ts[(Ns - len(yte)):Ns]
tga = ts[len(ytr):(Ns - len(yte))]
yga = y[len(ytr):(Ns - len(yte))]


# plot the results
plt.plot(ttr, ytr, '.', label ="training")
plt.plot(tga, yga, '.', label ="gap")
plt.plot(tte, yte, '.', label ="test")
plt.plot(tte, yte_p, label ="predicted")

plt.xlabel("Segment Number")
plt.ylabel("Target")
plt.legend()
plt.show()

