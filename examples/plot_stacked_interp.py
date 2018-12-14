'''
===========================
Resampling Time Series Data
===========================
This is a basic example illustrating the resampling of stacked format time series data
This may be useful for resampling irregularly sampled time series, or for determining
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
    # number of samples in a 2 second period
    period = params['stacked_interp__sample_period']

    return int(2. / period)


def calc_segment_width_b(params):
    # number of samples in a 2 second period -- input data in nanoseconds
    period = params['stacked_interp__sample_period']/(inNanoseconds*10**9)
    return int(2. / period)


# Boolean: 1 if input data is in nanoseconds - 0 if not
inNanoseconds = 1

# seed RNGESUS
np.random.seed(123124)

# load the data
X = load_stacked_data()

print("data = " + str(X))

N = len(X)

# I am adding in a column to represent targets (y) since my data doesn't include it
y = [np.array(np.arange(len(X[i])) + np.random.rand(len(X[i]))).astype(float) for i in np.arange(N)]

# Define the sample period of the resampled data
sample_period = (1. / 100.)*(inNanoseconds*10**9)

print("X before fit = " + str(X))


#clf = Pype([('stacked_interp', Stacked_Interp(sample_period, categorical_target=True)), changed categorical target to false 
clf = Pype([('stacked_interp', StackedInterp(sample_period, categorical_target=False)),
            #('segment', SegmentXY(width=100)),
            ('segment', SegmentXY(width=100)),
            ('features', FeatureRep()),
            ('scaler', StandardScaler()),
            #('rf', RandomForestClassifier(n_estimators=20))])
            ('rf', RandomForestRegressor(n_estimators=20))])

print("Number of time series before fit = " + str(len(X)))

# split the data
#for
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

print("Number of time series train = " + str(len(X_train)))

clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)

print("N series in train: ", len(X_train))
print("N series in test: ", len(X_test))
print("N segments in train: ", clf.N_train)
print("N segments in test: ", clf.N_test)
print("Accuracy score: ", score)

# lets try a few different sampling periods
# temporal splitting of data
splitter = TemporalKFold(n_splits=3)
Xs, ys, cv = splitter.split(X, y)

# here we use a callable parameter to force the segmenter width to equal 2 seconds
# note this is an extension of the sklearn api for setting class parameters
par_grid = {'stacked_interp__sample_period': [(10. / 100.)*(inNanoseconds*10**9),
                                              (1. / 100.)*(inNanoseconds*10**9),
                                              (1. / 500.)*(inNanoseconds*10**9),
                                              (1. / 100.)*(inNanoseconds*10**9)],
            'segment__width': [calc_segment_width_b]}

clf = GridSearchCV(clf, par_grid, cv=cv)
clf.fit(Xs, ys)
scores = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']

plt.plot(par_grid['stacked_interp__sample_period'], scores, '-o')
plt.title("Grid Search Scores")
plt.xlabel("Sample Period [ns]")
plt.ylabel("CV Average Score")
plt.fill_between(par_grid['stacked_interp__sample_period'], scores - stds, scores + stds, alpha=0.2,
                 color='navy')
plt.show()