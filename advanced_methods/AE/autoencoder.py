# Taken from https://towardsdatascience.com/extreme-rare-event-classification-using-autoencoders-in-keras-a565b386f098
# With some modifications

import numpy as np

from pylab import rcParams
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from advanced_methods.AE.utils import build_ae_model

rcParams['figure.figsize'] = 8, 6
LABELS = ["Normal", "Break"]


class Autoencoder(object):
    def __init__(self, x_train, dataset_string, verbosity=0):
        self.x_train = x_train
        self.dataset_string = dataset_string
        self.verbosity = 1 if verbosity == 2 else 0

        self.input_dim = None
        self.nb_epoch = None
        self.batch_size = None
        self.split_pct = None
        self.learning_rate = None
        self.dims = None

        self.threshold = None
        self.autoencoder = None

    def set_parameters(self):
        self.input_dim = self.x_train.shape[1]
        self.nb_epoch = 50
        self.batch_size = 128
        self.split_pct = 0.2
        self.learning_rate = 1e-3

        if self.dataset_string == "paysim" or self.dataset_string == "paysim_custom":
            self.dims = [self.x_train.shape[1], 16, 8, 4]
        elif self.dataset_string == "ccfraud":
            self.dims = [self.x_train.shape[1], 32, 16, 8, 4]
        elif self.dataset_string == "ieee":
            self.dims = [self.x_train.shape[1], 512, 256, 64, 4]

    def build(self):
        autoencoder = build_ae_model(self.dims, self.learning_rate)

        autoencoder.compile(metrics=['accuracy'],
                            loss='mean_squared_error',
                            optimizer='adam')

        x_train_split, x_valid_split = train_test_split(self.x_train, test_size=self.split_pct)

        autoencoder.fit(x_train_split, x_train_split,
                        epochs=self.nb_epoch,
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
