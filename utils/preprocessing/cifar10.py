import numpy as np

class Preprocess_cifar10:
    def __init__(self, anomaly_number):
        self.anomaly_number = anomaly_number

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

    def initial_processing(self, images, labels):
        anomaly_number = self.anomaly_number

        def is_anomaly(x):
            return True if x == anomaly_number else False

        def is_not_anomaly(x):
            return True if x != anomaly_number else False

        x_ben = images[np.vectorize(is_not_anomaly)(labels)]
        x_fraud = images[np.vectorize(is_anomaly)(labels)]

        return x_ben, x_fraud
