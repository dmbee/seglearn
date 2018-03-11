# N = len(Xt)

        # if Xs is not None:
        #     arglist = [[Xt[i], Xs[i], list(self.features.values())] for i in range(N)]
        # else:
        #     arglist = [[Xt[i], None, list(self.features.values())] for i in range(N)]
        #
        # if self.multithread == True:
        #     N_threads = cpu_count()
        #     pool = Pool(N_threads)
        #     Xts = np.split(Xt, N_threads)
        #
        #
        #     Xss = np.split(Xs, N_threads)
        #
        #
        #     X_new = pool.map(_feature_thread, arglist)
        # else:
        #     X_new = []
        #     for i in range(N):
        #         X_new.append(_feature_thread(arglist[i]))
def _feature_thread(args):
    ''' helper function for threading '''
    return _compute_features(*args)

def _compute_features(Xti, Xsi, features):
    '''
    Computes features for a segmented time series instance

    Parameters
    ----------
    Xti : array-like shape [n_segments, width, n_variables]
        segmented time series instance
    Xsi : array-like [n_static_variables]
        static variables associated with time series instance
    features :
        feature function dictionary

    Returns
    -------
    fts : array-like shape [n_segments, n_features]
        feature representation of Xti and Xsi
    '''
    N = Xti.shape[0]
    # computed features
    fts = [features[i](Xti) for i in range(len(features))]
    # static features
    s_fts = []
    if Xsi is not None:
        Ns = len(np.atleast_1d(Xsi))
        s_fts = [np.full((N,Ns), Xsi)]
    fts = np.column_stack(fts+s_fts)

    return fts

