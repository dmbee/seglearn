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
from seglearn.base import TS_Data

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import f1_score, make_scorer

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# seed RNGESUS
np.random.seed(123124)

# load the data
data = load_watch()

X = data['X']
y = data['y']

# I am adding in a column to represent time (50 Hz sampling), since my data doesn't include it
# the Interp class assumes time is the first column in the series
X = np.array([np.column_stack([np.arange(len(X[i]))/50.,X[i]]) for i in np.arange(len(X))])

clf = Pype([('interp', Interp(1/.25, categorical_target=True)),
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

# sampling frequency vs accuracy plot

img = mpimg.imread('feet.jpg')
plt.imshow(img)