import numpy as np

from mlxtend.data import loadlocal_mnist

from utils.preprocessing.mnist import Preprocess_mnist


def get_data_mnist(path, anomaly_number):
    pp_mnist = Preprocess_mnist()

    train_images, train_labels = loadlocal_mnist(
        images_path=path['train_images'],
        labels_path=path['train_labels'])
    test_images, test_labels = loadlocal_mnist(
        images_path=path['test_images'],
        labels_path=path['test_labels'])

    def is_anomaly(x):
        return True if x == anomaly_number else False

    def is_not_anomaly(x):
        return True if x != anomaly_number else False

    train_benign = train_images[np.vectorize(is_not_anomaly)(train_labels)]
    train_fraud = train_images[np.vectorize(is_anomaly)(train_labels)]

    test_benign = test_images[np.vectorize(is_not_anomaly)(test_labels)]
    test_fraud = test_images[np.vectorize(is_anomaly)(test_labels)]

    train_test_dimensions = {
        'train_benign': train_benign.shape[0],
        'train_fraud': train_fraud.shape[0],
        'test_benign': test_benign.shape[0],
        'test_fraud': test_fraud.shape[0],
    }

    pp_mnist.set_train_test_dimensions(train_test_dimensions)

    x_ben = np.concatenate((train_benign, test_benign))
    x_fraud = np.concatenate((train_fraud, test_fraud))

    return x_ben, x_fraud, pp_mnist

