# Taken from https://keras.io/examples/variational_autoencoder/ and
# https://link.springer.com/content/pdf/10.1007%2F978-1-4842-5177-5.pdf with some modifications

import numpy as np
from sklearn import metrics

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_recall_curve, roc_auc_score, \
    confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow_core.python.keras import Input, Model
from tensorflow_core.python.keras.layers import Dense, Lambda
from tensorflow_core.python.keras.losses import mse
import tensorflow.keras.backend as K

from advanced_methods.VAE.utils import sampling


class VAE(object):
    def __init__(self, x_train, dataset_string, seed, verbosity=0):
        self.label = 'VAE'

        self.x_train = x_train
        self.dataset_string = dataset_string
        self.seed = seed
        self.verbosity = 1 if verbosity == 2 else 0

        self.epochs = None
        self.batch_size = None
        self.original_dim = None
        self.input_shape = None
        self.intermediate_dim = None
        self.latent_dim = None
        self.activation_fct = None
        self.optimizer = None
        self.loss = None
        self.train_test_split = None

        self.threshold = None
        self.vae = None
        self.mse = None
        self.cm = None

    def set_parameters(self, parameters):
        self.original_dim = self.x_train.shape[1]
        self.input_shape = (self.original_dim,)

        self.epochs = parameters['epochs']
        self.batch_size = parameters['batch_size']
        self.intermediate_dim = parameters['intermediate_dim']
        self.latent_dim = parameters['latent_dim']
        self.activation_fct = parameters['activation_fct']
        self.optimizer = parameters['optimizer']
        self.loss = parameters['loss']
        self.train_test_split = parameters['train_test_split']

    def build(self):
        inputs = Input(shape=self.input_shape, name='encoder_input')
        x = Dense(self.intermediate_dim, activation=self.activation_fct)(inputs)
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

        # build decoder model
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        x = Dense(self.intermediate_dim, activation=self.activation_fct)(latent_inputs)
        outputs = Dense(self.original_dim, activation='sigmoid')(x)

        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')

        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae_mlp')

        # VAE Loss = mse_loss or xent_loss + kl_loss
        reconstruction_loss = mse(inputs, outputs)

        reconstruction_loss *= self.original_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)

        vae.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])

        x_train_split, x_valid_split = train_test_split(self.x_train, test_size=self.train_test_split,
                                                        random_state=self.seed)

        vae.fit(x_train_split, x_train_split, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbosity,
                shuffle=True, validation_data=(x_valid_split, x_valid_split))

        x_train_pred = vae.predict(self.x_train)
        train_mse = np.mean(np.power(self.x_train - x_train_pred, 2), axis=1)
        self.threshold = np.quantile(train_mse, 0.9)
        self.vae = vae

    def predict(self, x_test, y_test):
        y_pred = self.vae.predict(x_test)

        mse_ = np.mean(np.power(x_test - y_pred, 2), axis=1)
        self.mse = mse_
        y_pred = [1 if val > self.threshold else 0 for val in mse_]
        acc_score = accuracy_score(y_test, y_pred)

        precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, zero_division=0)
        self.cm = confusion_matrix(y_test, y_pred)

        precision_pts, recall_pts, _ = precision_recall_curve(y_test, mse_)
        pr_auc = metrics.auc(recall_pts, precision_pts)
        roc_auc = roc_auc_score(y_test, mse_)

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
        reconstructed_x_test = self.vae.predict(x_test)
        image_creator.add_image_plots(x_test, reconstructed_x_test, self.label, self.dataset_string, 10)

    def plot_conf_matrix(self, image_creator):
        image_creator.plot_conf_matrix(self.cm, self.label)
