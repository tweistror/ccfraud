# https://keras.io/examples/variational_autoencoder/
# https://link.springer.com/content/pdf/10.1007%2F978-1-4842-5177-5.pdf

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse
from keras import backend as K

import numpy as np

from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def execute_vae(x_train, x_test, y_test):
    original_dim = x_train.shape[1]

    # network parameters
    input_shape = (original_dim, )
    intermediate_dim = 12
    batch_size = 32
    latent_dim = 2
    epochs = 20

    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(original_dim, activation='sigmoid')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')

    # VAE Loss = mse_loss or xent_loss + kl_loss
    reconstruction_loss = mse(inputs, outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    vae.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    x_train_split, x_valid_split = train_test_split(x_train, test_size=0.2)

    vae.fit(x_train_split, x_train_split, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True,
            validation_data=(x_valid_split, x_valid_split))

    x_train_pred = vae.predict(x_train)
    train_mse = np.mean(np.power(x_train - x_train_pred, 2), axis=1)
    threshold = np.quantile(train_mse, 0.9)

    y_pred = vae.predict(x_test)
    test_mse = np.mean(np.power(x_test - y_pred, 2), axis=1)
    y_pred = [1 if val > threshold else 0 for val in test_mse]
    auc_score = roc_auc_score(y_test, y_pred)

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, zero_division=0)

    return precision[1], recall[1], fscore[1], auc_score, 'VAE'

