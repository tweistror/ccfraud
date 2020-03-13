import itertools

from keras import Input, regularizers, Model
from keras.layers import Dense


def build_ae_model(dims, learning_rate=1e-3, activation="relu"):
    input_layer = Input(shape=(dims[0],))

    encoder = Dense(dims[1], activation=activation, activity_regularizer=regularizers.l1(learning_rate))(input_layer)

    for dim in itertools.islice(dims, 2, None):
        encoder = Dense(dim, activation=activation)(encoder)

    dims_length = len(dims)
    for i in range(dims_length):
        if i == 0:
            decoder = Dense(dims[dims_length - 1 - i], activation=activation)(encoder)
        elif i == dims_length - 1:
            decoder = Dense(dims[dims_length - 1 - i], activation="linear")(decoder)
        else:
            decoder = Dense(dims[dims_length - 1 - i], activation=activation)(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)

    return autoencoder
