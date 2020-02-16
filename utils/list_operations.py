import numpy as np


def sample_shuffle(X):
    n_samples = len(X)
    s = np.arange(n_samples)
    np.random.shuffle(s)
    return np.array(X[s])


def clean_inf_nan(nparr):
    col_mean = np.nanmean(nparr, axis=0)
    indices = np.where(np.isnan(nparr))
    nparr[indices] = np.take(col_mean, indices[1])
    return nparr
