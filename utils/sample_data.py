import numpy as np

from utils.list_operations import sample_shuffle


def sample_data_for_unsupervised_baselines(x_ben, x_fraud, usv_train):
    n_test = len(x_fraud)

    x_ben, x_fraud = sample_shuffle(x_ben), sample_shuffle(x_fraud)

    x_train = x_ben[0:usv_train]
    x_test = x_ben[-n_test:].tolist() + x_fraud[-n_test:].tolist()

    x_test = np.array(x_test)

    y_test = np.zeros(2 * n_test)
    y_test[n_test:] = 1

    return x_train, x_test, y_test


def sample_data_for_supervised_baselines(x_ben, x_fraud, train_size, negative_samples):
    # Calculate dimensions
    n_train_benign = train_size - negative_samples
    n_train_fraud = train_size - n_train_benign

    n_test = len(x_fraud)

    # Shuffle
    x_ben, x_fraud = sample_shuffle(x_ben), sample_shuffle(x_fraud)

    # Testing data
    x_test = x_ben[-n_test:].tolist() + x_fraud[-n_test:].tolist()
    x_test = np.array(x_test)
    y_test = np.zeros(2 * n_test)
    y_test[n_test:] = 1

    # Select correct number of benign and fraud transactions
    x_ben = x_ben[0:n_train_benign]
    x_fraud = x_fraud[0:n_train_fraud]

    # Add labels
    x_ben = np.append(x_ben, np.zeros((len(x_ben), 1)), axis=1)
    x_fraud = np.append(x_fraud, np.ones((len(x_fraud), 1)), axis=1)

    # Create train and test sets
    train_data = np.concatenate((x_ben[0:n_train_benign], x_fraud[0:n_train_fraud]))

    train_data = sample_shuffle(train_data)

    x_train = train_data[:, :-1]
    y_train = train_data[:, -1]

    return x_train, x_test, y_train, y_test
