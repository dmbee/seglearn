'''
====================================
Simple FunctionXYTransformer Example
====================================

This example demonstrates how to execute arbitrary functions on time series data using the
FunctionXYTransformer.

'''

# Author: Matthias Gazzari
# License: BSD

from seglearn.transform import FunctionXYTransformer, SegmentXY
from seglearn.base import TS_Data

import numpy as np

def choose_cols(Xt, yt, cols):
	return [time_series[:, cols] for time_series in Xt], yt

def invert_y(Xt, yt):
	return Xt, np.logical_not(yt)

# Multivariate time series with 4 samples of 3 variables
X = [
	np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]),
	np.array([[30, 40, 50], [60, 70, 80], [90, 100, 110]]),
]
# Time series target
y = [
	np.array([True, False, False, True]),
	np.array([False, True, False]),
]

trans = FunctionXYTransformer(choose_cols, func_kwargs={"cols":[0,1]})
X, y, _ = trans.fit_transform(X, y)

segment = SegmentXY(width=3, overlap=1)
X, y, _ = segment.fit_transform(X, y)

print("X:", X)
print("y: ", y)

trans = FunctionXYTransformer(invert_y)
X, y, _ = trans.fit_transform(X, y)

print("~y: ", y)
