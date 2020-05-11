import numpy as np

from mlxtend.data import loadlocal_mnist

from utils.list_operations import sample_shuffle


def get_data_mnist(path, seed, fraud_number):
    train_images, train_labels = loadlocal_mnist(
        images_path=path['train_images'],
        labels_path=path['train_labels'])
    test_images, test_labels = loadlocal_mnist(
        images_path=path['test_images'],
        labels_path=path['test_labels'])

    def is_label(x):
        return True if x != fraud_number else False

    def is_not_label(x):
        return True if x == fraud_number else False

    train_benign = train_images[np.vectorize(is_label)(train_labels)]
    train_fraud = train_images[np.vectorize(is_not_label)(train_labels)]

    return sample_shuffle(train_benign, seed), sample_shuffle(train_fraud, seed)

