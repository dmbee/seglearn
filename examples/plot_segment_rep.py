'''
===================================================
Classifying Segments Directly with a Neural Network
===================================================

This is a basic example using a convolutional recurrent neural network to learn segments directly from time series data

'''
# Author: David Burns
# License: BSD

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.layers import Dense, LSTM, Conv1D
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split

from seglearn.datasets import load_watch
from seglearn.pipe import Pype
from seglearn.transform import SegmentX


def crnn_model(width=100, n_vars=6, n_classes=7, conv_kernel_size=5,
               conv_filters=64, lstm_units=100):
    input_shape = (width, n_vars)
    model = Sequential()
    model.add(Conv1D(filters=conv_filters, kernel_size=conv_kernel_size,
                     padding='valid', activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=conv_filters, kernel_size=conv_kernel_size,
                     padding='valid', activation='relu'))
    model.add(LSTM(units=lstm_units, dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(n_classes, activation="softmax"))

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    return model


# load the data
data = load_watch()
X = data['X']
y = data['y']

# create a segment learning pipeline
width = 100

pipe = Pype([('seg', SegmentX()),
             ('crnn', KerasClassifier(build_fn=crnn_model, epochs=10, batch_size=256, verbose=0))])

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)

print("N series in train: ", len(X_train))
print("N series in test: ", len(X_test))
print("N segments in train: ", pipe.N_train)
print("N segments in test: ", pipe.N_test)
print("Accuracy score: ", score)

img = mpimg.imread('segments.jpg')
plt.imshow(img)
