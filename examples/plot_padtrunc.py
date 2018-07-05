'''
=================================================
Pipeline using Time Series Padding and Truncation
=================================================

This is a basic example using the pipeline to learn a feature representation of the time series data
using padding and truncation instead of sliding window segmentation.

'''
# Author: David Burns
# License: BSD


from seglearn.transform import FeatureRep, PadTrunc
from seglearn.pipe import SegPipe
from seglearn.datasets import load_watch

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import numpy as np

# load the data
data = load_watch()
X = data['X']
y = data['y']

# create a feature representation pipeline with PadTrunc segmentation
# the time series are between 20-40 seconds
# this truncates them all to the first 5 seconds (sampling rate is 50 Hz)
truncator = PadTrunc(width = 250)

estimator = Pipeline([('features', FeatureRep()),
                 ('scaler', StandardScaler()),
                 ('svc', LinearSVC())])

# although optional, this shows how to set the RNG seed for SegPipe using random_state
pipe = SegPipe(est = estimator, segmenter = truncator, shuffle=True, random_state=42)

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=42)

pipe.fit(X_train,y_train)
score = pipe.score(X_test, y_test)

print("N series in train: ", len(X_train))
print("N series in test: ", len(X_test))
print("N segments in train: ", pipe.N_train)
print("N segments in test: ", pipe.N_test)
print("Accuracy score: ", score)

img = mpimg.imread('trunk.jpg')
plt.imshow(img)