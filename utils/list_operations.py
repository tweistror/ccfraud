import numpy as np


def sample_shuffle(x, seed):
    n_samples = len(x)
    s = np.arange(n_samples)
    np.random.shuffle(s)
    return np.array(x[s])


def clean_inf_nan(nparr):
    col_mean = np.nanmean(nparr, axis=0)
    indices = np.where(np.isnan(nparr))
    nparr[indices] = np.take(col_mean, indices[1])
    return nparr


def one_hot(x, depth):
    x_one_hot = np.zeros((len(x), depth), dtype=np.int32)
    x = x.astype(int)
    for i in range(x_one_hot.shape[0]):
        x_one_hot[i, x[i]] = 1
    return x_one_hot
