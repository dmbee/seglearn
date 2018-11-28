'''
=========================================
Simple SegmentedColumnTransformer Example
=========================================

This example demonstrates how to use the SegmentedColumnTransformer on segmented data.
Note that contextual data is not supported.
'''

# Author: Matthias Gazzari
# License: BSD

from seglearn.transform import SegmentXY, FeatureRep, SegmentedColumnTransformer
from seglearn.feature_functions import minimum
from seglearn.base import TS_Data

import numpy as np

X = [np.array([[0,1], [2,3], [4,5], [6,7], [8,9], [10,11], [12,13], [14,15]])]
y = [np.array([True, False, False, True, False, True, False, True])]

segment = SegmentXY(width=4, overlap=1)
X, y, _ = segment.fit_transform(X, y)

print('After segmentation:')
print(X, X.shape)
print(y, y.shape)

col_trans = SegmentedColumnTransformer([
    ('a', FeatureRep(features={'min_0': minimum}), 0),
    ('b', FeatureRep(features={'min_1': minimum}), 1),
    ('c', FeatureRep(features={'min_all': minimum}), [0,1]),
    # alternative column specifications:
    #('c', FeatureRep(features={'min_all': minimum}), lambda x: [0,1]),
    #('c', FeatureRep(features={'min_all': minimum}), slice(0,2)),
    #('c', FeatureRep(features={'min_all': minimum}), [True, True]),
])

print('After column-wise feature extraction:')
X = col_trans.fit_transform(X, y)
print(X, X.shape)
