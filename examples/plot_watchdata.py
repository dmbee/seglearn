'''
=============================
Working with Time Series Data
=============================

This example shows how to load the included smartwatch inertial sensor dataset, and create time series data objects compatible with the `seglearn` pipeline.

'''
# Author: David Burns
# License: BSD

from seglearn.datasets import load_watch
from seglearn.util import make_ts_data, check_ts_data, ts_stats, get_ts_data_parts

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = load_watch()

y = data['y']
Xt = data['X']
fs = 50 # sampling frequency

# create time series data object with no contextual variables
X = make_ts_data(Xt)
check_ts_data(X)

# create time series data object with 2 contextual variables
Xs = np.column_stack([data['side'], data['subject']])
X = make_ts_data(Xt, Xs)
check_ts_data(X)

# recover time series and contextual variables
Xt, Xs = get_ts_data_parts(X)

# generate some statistics from the time series data
results = ts_stats(X, y, fs = fs, class_labels = data['y_labels'])
print("DATA STATS - AGGREGATED")
print(results['total'])
print("")
print("DATA STATS - BY CLASS")
print(pd.DataFrame(results['by_class']))

# plot an instance from the data set
# this plot shows 6-axis inertial sensor data recorded by someone doing shoulder pendulum exercise
Xt0 = Xt[0]
f, axes = plt.subplots(nrows=1, ncols=2)
t=np.arange(len(Xt0)) / fs
axes[0].plot(t, Xt0[:,0], 'r-')
axes[0].plot(t, Xt0[:,1], 'g-')
axes[0].plot(t, Xt0[:,2], 'b-')
axes[0].set_xlabel('time [s]')
axes[0].set_ylabel('Acceleration [G]')
axes[0].legend(data['X_labels'][0:3])

axes[1].plot(t, Xt0[:,3], 'r-')
axes[1].plot(t, Xt0[:,4], 'g-')
axes[1].plot(t, Xt0[:,5], 'b-')
axes[1].set_xlabel('time [s]')
axes[1].set_ylabel('Rotational Velocity [rad/s]')
axes[1].legend(data['X_labels'][3:6])

plt.tight_layout()
plt.show()
