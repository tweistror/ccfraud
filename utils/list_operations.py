import numpy as np


def sample_shuffle(x, seed):
    n_samples = len(x)
    s = np.arange(n_samples)
    np.random.seed(seed)
    np.random.shuffle(s)
    return np.array(x[s])


def clean_inf_nan(nparr):
    col_mean = np.nanmean(nparr, axis=0)
    indices = np.where(np.isnan(nparr))
    nparr[indices] = np.take(col_mean, indices[1])
    return nparr
