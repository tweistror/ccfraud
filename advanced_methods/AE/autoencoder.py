# Taken from https://towardsdatascience.com/extreme-rare-event-classification-using-autoencoders-in-keras-a565b386f098
# With some modifications

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from advanced_methods.AE.utils import build_ae_model


class Autoencoder(object):
    def __init__(self, x_train, dataset_string, seed, verbosity=0):
        self.x_train = x_train
        self.dataset_string = dataset_string
        self.seed = seed
        self.verbosity = 1 if verbosity == 2 else 0

        self.input_dim = None
        self.epochs = None
        self.batch_size = None
        self.train_test_split = None
        self.learning_rate = None
        self.dims = None
        self.activation_fct = None

        self.threshold = None
        self.autoencoder = None

    def set_parameters(self, parameters):
        self.input_dim = self.x_train.shape[1]
        self.epochs = parameters['epochs']
        self.batch_size = parameters['batch_size']
        self.train_test_split = parameters['train_test_split']
        self.learning_rate = parameters['learning_rate']
        self.activation_fct = parameters['activation_fct']

        dim_input = self.x_train.shape[1]
        self.dims = parameters['dims']
        self.dims[0] = dim_input

    def build(self):
        autoencoder = build_ae_model(self.dims, self.learning_rate, self.activation_fct)

        autoencoder.compile(metrics=['accuracy'],
                            loss='mean_squared_error',
                            optimizer='adam')

        x_train_split, x_valid_split = train_test_split(self.x_train, test_size=self.train_test_split,
                                                        random_state=self.seed)

        autoencoder.fit(x_train_split, x_train_split,
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        shuffle=True,
                        validation_data=(x_valid_split, x_valid_split),
                        verbose=self.verbosity)

        x_train_pred = autoencoder.predict(self.x_train)
        mse = np.mean(np.power(self.x_train - x_train_pred, 2), axis=1)

        # Semi-supervised due to given threshold
        self.threshold = np.quantile(mse, 0.9)
        self.autoencoder = autoencoder

    def predict(self, x_test, y_test):
        # Predict the test set
        y_pred = self.autoencoder.predict(x_test)
        mse = np.mean(np.power(x_test - y_pred, 2), axis=1)
        y_pred = [1 if val > self.threshold else 0 for val in mse]
        acc_score = accuracy_score(y_test, y_pred)

        precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, zero_division=0)
        # class_report = classification_report(self.y_test, y_pred, target_names=['benign', 'fraud'], digits=4)

        return precision[1], recall[1], fscore[1], acc_score, 'Autoencoder'
