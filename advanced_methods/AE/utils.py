import itertools

from keras import Input, regularizers, Model
from keras.layers import Dense


def build_ae_model(dims, learning_rate, act_fct="relu", output_act_fct="sigmoid",
                   kernel_regularizer=None, activity_regularizer=None):

    activity_regularizer, kernel_regularizer = \
        get_ae_regularizers(learning_rate, activity_regularizer, kernel_regularizer)

    input_layer = Input(shape=(dims[0],))

    encoder = Dense(dims[1], activation=act_fct, activity_regularizer=activity_regularizer,
                    kernel_regularizer=kernel_regularizer)(input_layer)

    for dim in itertools.islice(dims, 2, None):
        encoder = Dense(dim, activation=act_fct)(encoder)

    dims_length = len(dims)
    for i in range(1, dims_length):
        # First Decoder Layer
        if i == 1:
            decoder = Dense(dims[dims_length - 1 - i], activation=act_fct)(encoder)
        # Last Decoder Layer
        elif i == dims_length - 1:
            decoder = Dense(dims[dims_length - 1 - i], activation=output_act_fct)(decoder)
        else:
            decoder = Dense(dims[dims_length - 1 - i], activation=act_fct)(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)

    return autoencoder


def get_ae_regularizers(learning_rate, activity_regularizer, kernel_regularizer):
    if activity_regularizer == 'l1':
        activity_regularizer = regularizers.l1(learning_rate)
    elif activity_regularizer == 'l2':
        activity_regularizer = regularizers.l2(learning_rate)
    else:
        activity_regularizer = None

    if kernel_regularizer == 'l1':
        kernel_regularizer = regularizers.l1(learning_rate)
    elif kernel_regularizer == 'l2':
        kernel_regularizer = regularizers.l2(learning_rate)
    else:
        kernel_regularizer = None

    return activity_regularizer, kernel_regularizer
