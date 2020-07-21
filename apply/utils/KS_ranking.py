import numpy as np

def KS_ranking(S, alfa=0.05):
    if isinstance(S, np.ndarray):
        if S.dtype.kind != 'f' and S.dtype.kind != 'i' and S.dtype.kind != 'u':
            raise ValueError('Elements in "S" must be int or float.')
        Ns = S.shape
        R = len(S) # number of sample sizes
        # S = [S] # create list to simply computation
        if len(Ns) != 2:
            raise ValueError('"S" must be of shape (Nboot,M) where Nboot>=1 and M>=1.')

    elif isinstance(S, list):
        if not all(isinstance(i, np.ndarray) for i in S):
            raise ValueError('Elements in "S" must be int or float.')
        Ns = S[0].shape
        R = len(S) # number of sample sizes
        if len(Ns) != 2:
            raise ValueError('"S[i]" must be of shape (Nboot,M) where Nboot>=1 and M>=1.')
    else:
        raise ValueError('"S" must be a list of a numpy.ndarray.')

    M = Ns[1]
    Nboot = Ns[0]

    ###########################################################################
    # Check optional inputs
    ###########################################################################
    if not isinstance(alfa, (float, np.float16, np.float32, np.float64)):
        raise ValueError('"alfa" must be scalar and numeric.')
    if alfa < 0 or alfa > 1:
        raise ValueError('"alfa" must be in (0,1).')

    ###########################################################################
    # Compute statistics across bootstrap resamples
    ###########################################################################
    # Variable initialization
    rank_m = np.nan*np.ones((Nboot, M))
    rank_conf = np.nan*np.ones((2, M))

    for j in range(Nboot): # loop over sample sizes

        rank_m[j, :] = np.argsort(S[j]).argsort() # bootstrap mean


    rank_conf = np.quantile(rank_m,[alfa/2, 1 - (alfa/2)], axis=0) # lower bound
    rank_ave = np.mean(rank_m, axis=0)

    return rank_ave, rank_conf
