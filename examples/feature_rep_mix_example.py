'''
============================
Simple FeatureRepMix Example
============================

This example demonstrates how to use the FeatureRepMix on segmented data.
'''

# Author: Matthias Gazzari
# License: BSD

from seglearn.transform import SegmentXY, FeatureRep, FeatureRepMix
from seglearn.feature_functions import minimum, maximum
from seglearn.base import TS_Data

import numpy as np
import pandas as pd

X = [np.array([[0,1,2,3], [4,5,6,7], [8,9,10,11]])]
y = [np.array([True, False, False])]

segment = SegmentXY(width=3, overlap=1)
X, y, _ = segment.fit_transform(X, y)

print('After segmentation:')
print(X, X.shape)
print(y, y.shape)

union = FeatureRepMix([
    ('a', FeatureRep(features={'min': minimum}), 0),
    ('b', FeatureRep(features={'min': minimum}), 1),
    ('c', FeatureRep(features={'min': minimum}), [2,3]),
    ('d', FeatureRep(features={'max': maximum}), slice(0,2)),
    ('e', FeatureRep(features={'max': maximum}), [False, False, True, True]),
])

X = union.fit_transform(X, y)
print('After column-wise feature extraction:')
df = pd.DataFrame(data=X, columns=union.f_labels)
print(df)
