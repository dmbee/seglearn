'''
==========================
Hyperparameter Selection 2
==========================

This example demonstrates how to do model selection in a pipeline where segments are learned directly by a neural network
'''

# Author: David Burns
# License: BSD

from seglearn.pipe import SegPipe
from seglearn.datasets import load_watch
from seglearn.split import TemporalKFold

from sklearn.model_selection import GridSearchCV
from keras.layers import Dense, LSTM, Conv1D
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier

import matplotlib.pyplot as plt

def crnn_model(width=100, n_vars=6, n_classes=7, conv_kernel_size=5,
               conv_filters=64, lstm_units=100):
    # create a crnn model with keras with two cnn layers, and one rnn layer
    input_shape = (width, n_vars)
    model = Sequential()
    model.add(Conv1D(filters=conv_filters, kernel_size=conv_kernel_size,
                     padding='valid', activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=conv_filters, kernel_size=conv_kernel_size,
                     padding='valid', activation='relu'))
    model.add(LSTM(units=lstm_units, dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(n_classes, activation="softmax"))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


# load the data
data = load_watch()
X = data['X']
y = data['y']

# temporal splitting of data
splitter = TemporalKFold(n_splits=3)
Xs, ys, cv = splitter.split(X,y)

# create a segment learning pipeline
width = 100
est = KerasClassifier(build_fn=crnn_model, epochs = 5, batch_size = 256, verbose = 0)
pipe = SegPipe(est)

# create a parameter dictionary using the SegPipe API - which is similar to the sklearn API
#
# parameters passed to an estimator in the ``feed`` pipeline are keyed ``f$estimator__parameter``
# parameters passed to an estimator in the ``est`` pipeline are keyed ``e$estimator__parameter``
#
# when the ``feed`` or ``est`` pipeline is not a pipeline, but just a single estimator
# the parameter would be keyed f$parameter or e$parameter respectively
#
# you can also set a parameter to be always equal to another parameter, by setting its value to
# parameter name to track
#
# note that if you want to set a parameter to a single value, it will still need to be as a list

par_grid = {'s$width' : [50,100,200],
            's$overlap' : [0.],
            'width' : ['s$width']}

clf = GridSearchCV(pipe, par_grid, cv=cv)
clf.fit(Xs, ys)
scores = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']

plt.plot(par_grid['s$width'], scores, '-o')
plt.title("Grid Search Scores")
plt.xlabel("Width [s]")
plt.ylabel("CV Average Score")
plt.fill_between(par_grid['s$width'], scores - stds, scores + stds, alpha=0.2, color = 'navy')
plt.show()
