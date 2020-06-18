import numpy as np

from utils.preprocessing.utils import unpickle


class Preprocess_cifar10:
    def __init__(self, path, anomaly_number, train_mode):
        self.path = path
        self.anomaly_number = anomaly_number
        self.train_mode = train_mode

        self.columns = None

        self.scaler = None
        self.pca = None
        self.train_test_dimensions = None
        self.scale_number = 255

    def set_train_test_dimensions(self, train_test_dimensions):
        self.train_test_dimensions = train_test_dimensions

    def preprocess(self, x_sv_train, x_usv_train, x_test):
        x_sv_train = x_sv_train.astype('float32')
        x_usv_train = x_usv_train.astype('float32')
        x_test = x_test.astype('float32')

        x_sv_train /= self.scale_number
        x_usv_train /= self.scale_number
        x_test /= self.scale_number

        return x_sv_train, x_usv_train, x_test

    def inverse_preprocessing(self, data):
        data = data.astype('float32')

        data *= self.scale_number

        return data

    def initial_processing(self):
        labels = None
        images = None

        for i in range(1, 7):
            batch_path = self.path[f'batch{i}']
            batch = unpickle(batch_path)
            if i == 1:
                labels = batch[b'labels']
                images = batch[b'data']
            else:
                labels = np.concatenate((labels, batch[b'labels']))
                images = np.concatenate((images, batch[b'data']))

        anomaly_number = self.anomaly_number

        def is_anomaly(x):
            if self.train_mode == 'rest':
                return True if x == anomaly_number else False
            return True if x != anomaly_number else False

        def is_not_anomaly(x):
            if self.train_mode == 'rest':
                return True if x != anomaly_number else False
            return True if x == anomaly_number else False

        x_ben = images[np.vectorize(is_not_anomaly)(labels)]
        x_fraud = images[np.vectorize(is_anomaly)(labels)]

        return x_ben, x_fraud


def get_cifar10_object(number):
    cifar10_dict = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck',
    }

    return cifar10_dict[number]
