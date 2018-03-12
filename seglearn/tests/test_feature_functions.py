from seglearn import feature_functions

import numpy as np

def test_mv_feature_functions():
    ''' test feature functions with multivariate data '''

    # sliding window data is shape [n_segments, width, variables]
    N = 20
    W = 30
    mv_data = np.random.rand(N, W, 3)

    ftr_funcs = feature_functions.all_features()

    for f in ftr_funcs:
        mvf = ftr_funcs[f](mv_data)
        assert len(mvf) == N



def test_uv_feature_functions():
    ''' test feature functions with univariate data '''
    N = 20
    W = 30
    uv_data = np.random.rand(N, W)

    ftr_funcs = feature_functions.all_features()

    for f in ftr_funcs:
        uvf = ftr_funcs[f](uv_data)
        assert len(uvf) == N


