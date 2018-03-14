'''
====================================================
Basic Feature Representation Classification Pipeline
====================================================

This is a basic example using the pipeline to learn a feature representation of the time series data

'''
# Author: David Burns
# License: BSD


from seglearn.transform import FeatureRep
from seglearn.pipe import SegPipe
from seglearn.datasets import load_watch
from seglearn.base import TS_Data

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import f1_score, make_scorer

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# load the data
data = load_watch()
X = data['X']
y = data['y']

# create a feature representation pipeline

est = Pipeline([('features', FeatureRep()),
                ('scaler', StandardScaler()),
                ('rf', RandomForestClassifier())])
pipe = SegPipe(est)

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

pipe.fit(X_train,y_train)
score = pipe.score(X_test, y_test)

print("N series in train: ", len(X_train))
print("N series in test: ", len(X_test))
print("N segments in train: ", pipe.N_train)
print("N segments in test: ", pipe.N_test)
print("Accuracy score: ", score)

# now lets add some contextual data
Xc = np.column_stack((data['side'], data['subject']))
Xt = np.array(data['X'])
X = TS_Data(Xt, Xc)
y = np.array(data['y'])

# and do a cross validation
scoring = make_scorer(f1_score, average = 'macro')
cv_scores = cross_validate(pipe, X, y, cv = 4, return_train_score=True)
print("CV Scores: ", pd.DataFrame(cv_scores))

# lets see what feature we used
print("Features: ", pipe.est.steps[0][1].f_labels)

img = mpimg.imread('feet.jpg')
plt.imshow(img)