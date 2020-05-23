import numpy as np


class Preprocess_mnist:
    def __init__(self, anomaly_number, train_mode):
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

    def initial_processing(self, train_images, train_labels, test_images, test_labels):
        anomaly_number = self.anomaly_number

        def is_anomaly(x):
            if self.train_mode == 'rest':
                return True if x == anomaly_number else False
            return True if x != anomaly_number else False

        def is_not_anomaly(x):
            if self.train_mode == 'rest':
                return True if x != anomaly_number else False
            return True if x == anomaly_number else False

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

        self.set_train_test_dimensions(train_test_dimensions)

        x_ben = np.concatenate((train_benign, test_benign))
        x_fraud = np.concatenate((train_fraud, test_fraud))

        return x_ben, x_fraud
