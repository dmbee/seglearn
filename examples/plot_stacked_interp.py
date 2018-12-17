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

from seglearn.datasets import load_stacked_data
from seglearn.transform import StackedInterp

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

# Matplotlib style parameters
left = 0.125  # the left side of the subplots of the figure
right = 0.9  # the right side of the subplots of the figure
bottom = 0.1  # the bottom of the subplots of the figure
top = 0.9  # the top of the subplots of the figure
wspace = 0.3  # the amount of width reserved for space between subplots,
# expressed as a fraction of the average axis width
hspace = 0.4  # the amount of height reserved for space between subplots,
# expressed as a fraction of the average axis height

plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

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

        # set the grid on
        axarr[i, j].grid('on')

        # adjust axis label font
        xlab = axarr[i, j].xaxis.get_label()
        ylab = axarr[i, j].yaxis.get_label()

        xlab.set_style('italic')
        xlab.set_size(8)
        ylab.set_style('italic')
        ylab.set_size(8)

        # adjust title font
        ttl = axarr[i, j].title
        ttl.set_weight('bold')
        ttl.set_size('8')

plt.show()