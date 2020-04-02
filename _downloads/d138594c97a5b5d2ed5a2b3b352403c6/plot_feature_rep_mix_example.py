"""
============================
Simple FeatureRepMix Example
============================

This example demonstrates how to use the FeatureRepMix on segmented data.

Although not shown here, FeatureRepMix can be used with Pype in place of FeatureRep.
See API documentation for an example.
"""

# Author: Matthias Gazzari
# License: BSD

from seglearn.transform import Segment, FeatureRep, FeatureRepMix
from seglearn.feature_functions import minimum, maximum
from seglearn.base import TS_Data

import numpy as np
import pandas as pd

# Single multivariate time series with 3 samples of 4 variables
X = [np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])]
# Time series target
y = [np.array([True, False, False])]

segment = Segment(width=3, overlap=1)
X, y, _ = segment.fit_transform(X, y)

print('After segmentation:')
print("X:", X)
print("y: ", y)

union = FeatureRepMix([
    ('a', FeatureRep(features={'min': minimum}), 0),
    ('b', FeatureRep(features={'min': minimum}), 1),
    ('c', FeatureRep(features={'min': minimum}), [2, 3]),
    ('d', FeatureRep(features={'max': maximum}), slice(0, 2)),
    ('e', FeatureRep(features={'max': maximum}), [False, False, True, True]),
])

X = union.fit_transform(X, y)
print('After column-wise feature extraction:')
df = pd.DataFrame(data=X, columns=union.f_labels)
print(df)
