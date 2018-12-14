'''
===========================
Resampling Stacked Time Series Data
===========================
This is a basic example illustrating the resampling of stacked format time series data
This may be useful for resampling irregularly sampled time series in stacked format, or for determining
an optimal sampling frequency for the data
'''
# Author: Phil Boyer
# License: BSD

import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from seglearn.datasets import load_stacked_data
from seglearn.pipe import Pype
from seglearn.split import TemporalKFold
from seglearn.transform import FeatureRep, SegmentXY, StackedInterp

def calc_segment_width(params):
    # number of samples in a 2 second period -- input data in nanoseconds
    period = params['stacked_interp__sample_period']/(inNanoseconds*10**9)
    return int(2. / period)


# Boolean: 1 if input data is in nanoseconds - 0 if not
inNanoseconds = 1

# seed RNGESUS
np.random.seed(123124)

# load the data
X = load_stacked_data()

#print("data = " + str(X))

N = len(X)

# I am adding in a column to represent targets (y) since my data doesn't include it
y = [np.array(np.arange(len(X[i])) + np.random.rand(len(X[i]))).astype(float) for i in np.arange(N)]

# Plot the 2 time series, each of the sensors versus the new sample period for 2 sample periods
sample_periods = [(10. / 100.)*(inNanoseconds*10**9), (1. / 100.)*(inNanoseconds*10**9), (0.1 / 100.)*(inNanoseconds*10**9)]

f, axarr = plt.subplots(2, 3)

for j in np.arange(len(sample_periods)):
    print("\nSample Period = " + str(sample_periods[j] / (inNanoseconds * 10 ** 9)) + " s")
    stacked_interp = StackedInterp(sample_periods[j])
    stacked_interp.fit(X, y)
    Xc, yc, swt = stacked_interp.transform(X, y)

    for i in np.arange(N):
        print("X[" + str(i) + "] length = " + str(len(X[i])))
        print("X[" + str(i) + "] length after interpolation to sample period = " + str(len(Xc[i])))
        axarr[i, j].plot(Xc[i])
        axarr[i, j].set_title("Interpolated Series " + str(i) + " : Sample Period = "
                              + str(sample_periods[j]/(inNanoseconds*10**9)) + " s")
        axarr[i, j].set_xlabel("Sample Number")
        axarr[i, j].set_ylabel("Value")

plt.show()