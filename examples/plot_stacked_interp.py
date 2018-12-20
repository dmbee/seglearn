'''
===========================
Resampling Stacked Time Series Data
===========================
This is a basic example illustrating the resampling of stacked format time series data.
This may be useful for resampling irregularly sampled time series in stacked format, or for determining
an optimal sampling frequency for the data
'''
# Author: Phil Boyer
# License: BSD

import numpy as np

from seglearn.transform import StackedInterp

# Simple stacked input with values from 2 sensors / 2 axis at irregular sample times

t = np.array([1.1, 1.2, 2.1, 3.3, 3.4, 3.5]).astype(float)
s = np.array([0, 1, 0, 0, 1, 1]).astype(float)
v1 = np.array([3, 4, 5, 7, 15, 25]).astype(float)
v2 = np.array([5, 7, 6, 9, 22, 35]).astype(float)
y = np.array([1, 2, 2, 2, 3, 3]).astype(float)
df = np.column_stack([t, s, v1, v2])

X = [df, df]
y = [y, y]

print("\nX input: \n" + str(X))

stacked_interp = StackedInterp(0.5)
stacked_interp.fit(X, y)
Xc, yc, swt = stacked_interp.transform(X, y)

print ("\nX interpolated: \n" + str(Xc))