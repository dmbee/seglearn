'''
==============================
Scoring Time Series Estimators
==============================

This examples demonstrates some of the caveats / issues when trying to
calculate performance scores for time series estimators.

This pipeline has been designed to evaluate performance using
segments (not series') as instances of the data.

'''
# Author: David Burns
# License: BSD


from seglearn.transform import FeatureRep, SegmentX
from seglearn.pipe import SegPipe
from seglearn.datasets import load_watch

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import f1_score, confusion_matrix, make_scorer

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools

##############################################
# CONFUSION PLOT
##############################################

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          cmap=plt.cm.Blues):
    ''' plots confusion matrix '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


##############################################
# SETUP
##############################################

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

##############################################
# OPTION 1: Use the score SegPipe score method
##############################################

pipe.fit(X_train,y_train)
score = pipe.score(X_test, y_test)
print("Accuracy score: ", score)

######################################################################
# OPTION 2: generate true and predicted target values for the segments
######################################################################

y_true, y_pred = pipe.predict(X_test, y_test)
# use any of the sklearn scorers
f1_macro = f1_score(y_true, y_pred, average='macro')
print("F1 score: ", f1_macro)

cm = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cm, data['y_labels'])

##########################################
# OPTION 3: scoring during model selection
##########################################

# model selection using the built-in score method for the final estimator
cv_scores = cross_validate(pipe, X, y, cv = 4, return_train_score=True)
print("CV Scores: ", pd.DataFrame(cv_scores))

# model selection with scoring functions / dictionaries
#
# unfortunately, this is not possible withing the current framework due to how
# scoring is implemented within the model_selection functions / classes of sklearn
# running the code below will cause an error, because the model_selection
# functions / classes do not have access to y_true for the segments
#
# >>> scoring = ['accuracy','precision_macro','recall_macro','f1_macro']
# >>> cv_scores = cross_validate(pipe, X, y, cv = 4, return_train_score=True, scoring=scoring)
#
# workarounds for this issue are outlined below


###################################################
# SCORING WORKAROUND 1: USE ANOTHER SCORER FUNCTION
###################################################

# ``SegPipe`` can be initialized with a scorer callable made with sklearn.metrics.make_scorer
# this can be used to cross_validate or grid search with any 1 score

scorer = make_scorer(f1_score, average='macro')
pipe = SegPipe(est, scorer = scorer)
cv_scores = cross_validate(pipe, X, y, cv = 4, return_train_score=True)
print("CV F1 Scores: ", pd.DataFrame(cv_scores))


##################################################
# SCORING WORKAROUND 2: WORK OUTSIDE THE PIPELINE
##################################################

# If you want to have multiple score computed, the only way is as follows
#
# First transform the time series data into segments and then score the ``est`` part of the
# pipeline.
#
# The disadvantage of this is that the parameters of the ``seg`` pipeline cannot be
# optimized with this approach

segmenter = SegmentX()
X_seg, y_seg, _ = segmenter.fit_transform(X, y)
scoring = ['accuracy','precision_macro','recall_macro','f1_macro']
cv_scores = cross_validate(est, X_seg, y_seg,
                           cv=4, return_train_score=False, scoring=scoring)
print("CV Scores (workaround): ", pd.DataFrame(cv_scores))

