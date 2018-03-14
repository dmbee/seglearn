'''
=======================
Time Series Forecasting
=======================

In this example, we use a feature representation pipeline to forecast a continuous time series
target with a regressor

'''
# Author: David Burns
# License: BSD


from seglearn.transform import FeatureRep, SegmentXY, mean
from seglearn.pipe import SegPipe
from seglearn.split import temporal_split, TemporalKFold

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import numpy as np

# remember for a single time series, we need to make it a list
X = [np.arange(5000)/100.]
y = [np.sin(X[0])*X[0]+X[0]*X[0]]

# split the data along the time axis (our only option since we have only 1 time series)
X_train, X_test, y_train, y_test = temporal_split(X, y)

# create a feature representation pipeline
est = Pipeline([('features', FeatureRep()),
                ('scaler', StandardScaler()),
                ('rf', Ridge())])

# setting y_func = mean, selects the mean from each y segment as the target
# forecast 8 with overlap 0.5 means we are predicting 4 segments ahead (200 time points)
# see the API documentation for further details
segmenter = SegmentXY(width = 50, overlap=0.5, y_func=mean, forecast=8)
pipe = SegPipe(est, segmenter)

# fit and score
pipe.fit(X_train,y_train)
score = pipe.score(X_test, y_test)

print("N series in train: ", len(X_train))
print("N series in test: ", len(X_test))
print("N segments in train: ", pipe.N_train)
print("N segments in test: ", pipe.N_test)
print("Score: ", score)

# lets plot the amazing results
ytr, ytrp = pipe.predict(X_train, y_train) # get the training targets
y, y_p = pipe.predict(X_test, y_test) # get test targets
xtr = np.arange(len(ytr)) # segment number
xte = np.arange(len(y)) + len(xtr)

# note there will be a gap due to discarded data in the test set
plt.plot(xtr, ytr, '.', label = "training")
plt.plot(xte, y, '.', label = "actual")
plt.plot(xte, y, label = "predicted")
plt.xlabel("Segment Number")
plt.ylabel("Target")
plt.legend()
plt.show()

