import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from pylab import rcParams
from keras.models import Model
from keras.layers import Input, Dense
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

# https://towardsdatascience.com/extreme-rare-event-classification-using-autoencoders-in-keras-a565b386f098

rcParams['figure.figsize'] = 8, 6
LABELS = ["Normal", "Break"]


class Autoencoder(object):
    def __init__(self, dataset_string, x_train, x_test, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_test = y_test
        self.dataset_string = dataset_string

    def build_network(self):
        input_dim = self.x_train.shape[1]
        learning_rate = 1e-3

        # TODO: FUnction building the layers based on lists
        if self.dataset_string == "paysim":
            # input_dim is 11
            encoding_dim = 16
            hidden_dim_1 = 8
            hidden_dim_2 = 4
            input_layer = Input(shape=(input_dim,))
            encoder = Dense(encoding_dim, activation="relu", activity_regularizer=regularizers.l1(learning_rate))(
                input_layer)
            encoder = Dense(hidden_dim_1, activation="relu")(encoder)
            encoder = Dense(hidden_dim_2, activation="relu")(encoder)
            decoder = Dense(hidden_dim_2, activation="relu")(encoder)
            decoder = Dense(hidden_dim_1, activation="relu")(decoder)
            decoder = Dense(encoding_dim, activation="relu")(decoder)
            decoder = Dense(input_dim, activation="linear")(decoder)
            autoencoder = Model(inputs=input_layer, outputs=decoder)
        elif self.dataset_string == "ccfraud":
            # input_dim is 28
            encoding_dim = 32
            hidden_dim_1 = 16
            hidden_dim_2 = 8
            hidden_dim_3 = 4
            input_layer = Input(shape=(input_dim,))
            encoder = Dense(encoding_dim, activation="relu", activity_regularizer=regularizers.l1(learning_rate))(
                input_layer)
            encoder = Dense(hidden_dim_1, activation="relu")(encoder)
            encoder = Dense(hidden_dim_2, activation="relu")(encoder)
            encoder = Dense(hidden_dim_3, activation="relu")(encoder)
            decoder = Dense(hidden_dim_3, activation="relu")(encoder)
            decoder = Dense(hidden_dim_2, activation="relu")(decoder)
            decoder = Dense(hidden_dim_1, activation="relu")(decoder)
            decoder = Dense(encoding_dim, activation="relu")(decoder)
            decoder = Dense(input_dim, activation="linear")(decoder)
            autoencoder = Model(inputs=input_layer, outputs=decoder)
        elif self.dataset_string == "ieee":
            # input_dim is 432
            encoding_dim = 512
            hidden_dim_1 = 256
            hidden_dim_2 = 128
            hidden_dim_3 = 64
            # hidden_dim_4 = 32
            input_layer = Input(shape=(input_dim,))
            encoder = Dense(encoding_dim, activation="relu", activity_regularizer=regularizers.l1(learning_rate))(
                input_layer)
            encoder = Dense(hidden_dim_1, activation="relu")(encoder)
            encoder = Dense(hidden_dim_2, activation="relu")(encoder)
            encoder = Dense(hidden_dim_3, activation="relu")(encoder)
            # encoder = Dense(hidden_dim_4, activation="relu")(encoder)

            # decoder = Dense(hidden_dim_4, activation="relu")(encoder)
            decoder = Dense(hidden_dim_3, activation="relu")(encoder)
            decoder = Dense(hidden_dim_2, activation="relu")(decoder)
            decoder = Dense(hidden_dim_1, activation="relu")(decoder)
            decoder = Dense(encoding_dim, activation="relu")(decoder)
            decoder = Dense(input_dim, activation="linear")(decoder)
            autoencoder = Model(inputs=input_layer, outputs=decoder)

        return autoencoder

    def execute_autoencoder(self):
        nb_epoch = 50
        batch_size = 128
        split_pct = 0.2
        autoencoder = self.build_network()

        autoencoder.compile(metrics=['accuracy'],
                            loss='mean_squared_error',
                            optimizer='adam')

        x_train_split, x_valid_split = train_test_split(self.x_train, test_size=split_pct)

        autoencoder.fit(x_train_split, x_train_split,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(x_valid_split, x_valid_split),
                        verbose=0)

        # TODO: How to set threshold? Clustering?
        # Predict the training set for setting the mse threshold
        x_train_pred = autoencoder.predict(self.x_train)
        mse = np.mean(np.power(self.x_train - x_train_pred, 2), axis=1)
        threshold = np.quantile(mse, 0.9)

        # error_df = pd.DataFrame({'Reconstruction_error': mse,
        #                          'True_class': np.zeros(self.x_train.shape[0])})
        # error_df = error_df.reset_index()
        # threshold_fixed = error_df['Reconstruction_error'].quantile(.9)
        # groups = error_df.groupby('True_class')
        # fig, ax = plt.subplots()
        # for name, group in groups:
        #     ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
        #             label="Break" if name == 1 else "Normal")
        # ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
        # ax.legend()
        # plt.title("Reconstruction error for different classes")
        # plt.ylabel("Reconstruction error")
        # plt.xlabel("Data point index")
        # plt.show()

        # Predict the test set
        y_pred = autoencoder.predict(self.x_test)
        mse = np.mean(np.power(self.x_test - y_pred, 2), axis=1)
        y_pred = [1 if val > threshold else 0 for val in mse]
        auc_score = roc_auc_score(self.y_test, y_pred)

        precision, recall, fscore, support = precision_recall_fscore_support(self.y_test, y_pred, zero_division=0)
        # class_report = classification_report(self.y_test, y_pred, target_names=['benign', 'fraud'], digits=4)

        # error_df_test = pd.DataFrame({'Reconstruction_error': mse,
        #                               'True_class': self.y_test})
        # error_df_test = error_df_test.reset_index()
        # groups = error_df_test.groupby('True_class')
        # fig, ax = plt.subplots()
        # for name, group in groups:
        #     ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
        #             label="Break" if name == 1 else "Normal")
        # ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
        # ax.legend()
        # plt.title("Reconstruction error for different classes")
        # plt.ylabel("Reconstruction error")
        # plt.xlabel("Data point index")
        # plt.show()

        # pred_y = [1 if e > threshold else 0 for e in error_df_test.Reconstruction_error.values]
        # conf_matrix = confusion_matrix(error_df_test.True_class, pred_y)
        # plt.figure(figsize=(12, 12))
        # sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
        # plt.title("Confusion matrix")
        # plt.ylabel('True class')
        # plt.xlabel('Predicted class')
        # plt.show()
        #
        # false_pos_rate, true_pos_rate, thresholds = roc_curve(error_df_test.True_class,
        #                                                       error_df_test.Reconstruction_error)
        # roc_auc = auc(false_pos_rate, true_pos_rate, )
        # plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f' % roc_auc)
        # plt.plot([0, 1], [0, 1], linewidth=5)
        # plt.xlim([-0.01, 1])
        # plt.ylim([0, 1.01])
        # plt.legend(loc='lower right')
        # plt.title('Receiver operating characteristic curve (ROC)')
        # plt.ylabel('True Positive Rate')
        # plt.xlabel('False Positive Rate')
        # plt.show()

        return precision[1], recall[1], fscore[1], auc_score, 'USV-Autoencoder'

