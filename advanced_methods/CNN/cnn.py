import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, roc_auc_score, accuracy_score, precision_recall_fscore_support, \
    confusion_matrix

from sklearn.model_selection import train_test_split
from tensorflow_core.python.keras import Input, Model
from tensorflow_core.python.keras.layers import Conv2D, MaxPooling2D, UpSampling2D


class ConvolutionalNN:
    def __init__(self, x_train, dataset_string, seed, verbosity=-0):
        self.x_train = x_train
        self.dataset_string = dataset_string
        self.seed = seed
        self.verbosity = 1 if verbosity == 2 else 0

        self.epochs = 10
        self.batch_size = 32
        self.train_test_split = None

        self.threshold = None
        self.cnn_autoencoder = None

        self.mse = None
        self.label = 'CNN'
        self.cm = None

    def set_parameters(self, parameters):
        self.train_test_split = parameters['train_test_split']

    def build(self):
        input_img = Input(shape=(28, 28, 1))

        cnn = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        cnn = MaxPooling2D((2, 2), padding='same')(cnn)
        cnn = Conv2D(32, (3, 3), activation='relu', padding='same')(cnn)
        cnn = MaxPooling2D((2, 2), padding='same')(cnn)
        cnn = Conv2D(32, (3, 3), activation='relu', padding='same')(cnn)
        encoded = MaxPooling2D((2, 2), padding='same')(cnn)

        cnn = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
        cnn = UpSampling2D((2, 2))(cnn)
        cnn = Conv2D(32, (3, 3), activation='relu', padding='same')(cnn)
        cnn = UpSampling2D((2, 2))(cnn)
        cnn = Conv2D(32, (3, 3), activation='relu')(cnn)
        cnn = UpSampling2D((2, 2))(cnn)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(cnn)

        cnn_autoencoder = Model(input_img, decoded)
        cnn_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

        x_train = self.x_train.reshape(-1, 28, 28, 1)

        x_train_split, x_valid_split = train_test_split(x_train, test_size=self.train_test_split,
                                                        random_state=self.seed)

        cnn_autoencoder.fit(x_train_split, x_train_split,
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            validation_data=(x_valid_split, x_valid_split),
                            verbose=self.verbosity)

        x_train_pred = cnn_autoencoder.predict(x_train)
        mse = np.mean(np.power(x_train - x_train_pred, 2), axis=1)

        # Semi-supervised due to given threshold
        self.threshold = np.quantile(mse, 0.9)
        self.cnn_autoencoder = cnn_autoencoder

    def predict(self, x_test, y_test):
        # Predict the test set
        x_test = x_test.reshape(-1, 28, 28, 1)

        y_pred = self.cnn_autoencoder.predict(x_test)

        x_test = x_test.reshape(-1, 28 * 28)
        y_pred = y_pred.reshape(-1, 28 * 28)

        mse = np.mean(np.power(x_test - y_pred, 2), axis=1)
        self.mse = mse

        precision_pts, recall_pts, _ = precision_recall_curve(y_test, mse)
        pr_auc = metrics.auc(recall_pts, precision_pts)
        roc_auc = roc_auc_score(y_test, mse)

        y_pred = [1 if val > self.threshold else 0 for val in mse]
        acc_score = accuracy_score(y_test, y_pred)

        precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, zero_division=0)
        self.cm = confusion_matrix(y_test, y_pred)
        # class_report = classification_report(self.y_test, y_pred, target_names=['benign', 'fraud'], digits=4)

        results = {
            'prec_list': [precision[1]],
            'reca_list': [recall[1]],
            'f1_list': [fscore[1]],
            'acc_list': [acc_score],
            'pr_auc_list': [pr_auc],
            'roc_auc_list': [roc_auc],
            'method_list': [self.label],
        }

        return results

    def build_plots(self, y_test, image_creator):
        image_creator.add_curves(y_test, self.mse, self.label)

    def plot_reconstructed_images(self, x_test, image_creator):
        x_test = x_test.reshape(-1, 28, 28, 1)
        reconstructed_x_test = self.cnn_autoencoder.predict(x_test)

        x_test = x_test.reshape(-1, 28 * 28)

        image_creator.add_image_plots(x_test, reconstructed_x_test, self.label, self.dataset_string, 10)

    def plot_conf_matrix(self, image_creator):
        image_creator.plot_conf_matrix(self.cm, self.label)
