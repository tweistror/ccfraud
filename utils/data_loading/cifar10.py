import pickle
import numpy as np

from utils.list_operations import sample_shuffle


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_data_cifar10(path, anomaly_number):
    batch1_path = path['batch1']
    batch2_path = path['batch2']
    batch3_path = path['batch2']

    train_batch1 = unpickle(batch1_path)
    train_batch2 = unpickle(batch2_path)
    train_batch3 = unpickle(batch3_path)

    labels = np.concatenate((train_batch1[b'labels'], train_batch2[b'labels'], train_batch3[b'labels']))
    images = np.concatenate((train_batch1[b'data'], train_batch2[b'data'], train_batch3[b'data']))

    def is_label(x):
        return True if x != anomaly_number else False

    def is_not_label(x):
        return True if x == anomaly_number else False

    train_benign = images[np.vectorize(is_not_label)(labels)]
    train_fraud = images[np.vectorize(is_label)(labels)]

    return train_benign, train_fraud
