import numpy as np

from utils.list_operations import sample_shuffle


def sample_data_for_occ(x_ben, x_fraud, dataset_string):
    # TODO: Handle different datasets

    n_training = len(x_ben)
    n_test = len(x_fraud)

    x_ben, x_fraud = sample_shuffle(x_ben), sample_shuffle(x_fraud)

    x_train = x_ben[0:n_training]
    x_test = x_ben[-n_test:].tolist() + x_fraud[-n_test:].tolist()

    x_test = np.array(x_test)

    y_train = np.zeros(n_training)
    y_test = np.zeros(2 * n_test)
    y_test[n_test:] = 1

    return x_train, x_test, y_train, y_test
