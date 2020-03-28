# Taken from https://keras.io/examples/variational_autoencoder/ and
# https://link.springer.com/content/pdf/10.1007%2F978-1-4842-5177-5.pdf with some modifications

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse
from keras import backend as K

import numpy as np

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from advanced_methods.VAE.utils import sampling


class VAE(object):
    def __init__(self, x_train, dataset_string, epochs=20, batch_size=32):
        self.x_train = x_train
        self.dataset_string = dataset_string

        self.epochs = epochs
        self.batch_size = batch_size

        self.original_dim = None
        self.input_shape = None
        self.intermediate_dim = None
        self.latent_dim = None

        self.threshold = None
        self.vae = None

    def set_parameters(self):
        self.original_dim = self.x_train.shape[1]
        self.input_shape = (self.original_dim,)

        if self.dataset_string == 'ccfraud':
            self.intermediate_dim = 12
            self.latent_dim = 2

        elif self.dataset_string == 'paysim':
            self.intermediate_dim = 5
            self.latent_dim = 2

        elif self.dataset_string == 'ieee':
            self.intermediate_dim = 200
            self.latent_dim = 2

    def build(self):
        inputs = Input(shape=self.input_shape, name='encoder_input')
        x = Dense(self.intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

        # build decoder model
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        x = Dense(self.intermediate_dim, activation='relu')(latent_inputs)
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

        vae.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

        x_train_split, x_valid_split = train_test_split(self.x_train, test_size=0.2)

        vae.fit(x_train_split, x_train_split, batch_size=self.batch_size, epochs=self.epochs, verbose=0, shuffle=True,
                validation_data=(x_valid_split, x_valid_split))

        x_train_pred = vae.predict(self.x_train)
        train_mse = np.mean(np.power(self.x_train - x_train_pred, 2), axis=1)
        self.threshold = np.quantile(train_mse, 0.9)
        self.vae = vae

    def predict(self, x_test, y_test):
        y_pred = self.vae.predict(x_test)
        test_mse = np.mean(np.power(x_test - y_pred, 2), axis=1)
        y_pred = [1 if val > self.threshold else 0 for val in test_mse]
        acc_score = accuracy_score(y_test, y_pred)

        precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, zero_division=0)

        return precision[1], recall[1], fscore[1], acc_score, 'VAE'
