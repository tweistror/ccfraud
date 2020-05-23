# Taken from https://github.com/PanpanZheng/OCAN with some modifications

import numpy as np
import theano.tensor as T

from keras.layers import Input, Dense, LSTM, RepeatVector
from keras.models import Sequential, Model
from keras.layers.core import Masking
from keras import regularizers


class Dense_Autoencoder(object):
    """docstring for LSTM_Autoencoder"""

    def __init__(self, input_dim, hidden_dim, epochs, verbosity=0):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.autoencoder = Autoencoder(epochs, 1 if verbosity == 2 else 0)
        self.autoencoder.modelMasking('dense', [self.input_dim], self.hidden_dim)

        self.hidden_representation = None

    def compile(self):
        self.autoencoder.compile()

    def fit(self, data):
        self.autoencoder.fit(data, 'nor')

    def get_hidden_layer(self):
        # print "net summary: ", self.autoencoder.model.summary()
        self.hidden_representation = Sequential()
        self.hidden_representation.add(self.autoencoder.model.layers[0])
        self.hidden_representation.add(self.autoencoder.model.layers[1])
        self.hidden_representation.add(self.autoencoder.model.layers[2])

    def get_hidden_representation(self, data):
        return self.hidden_representation.predict(data)


class Autoencoder(object):
    """docstring for Autoencoder"""

    # def __init__(self, sample_weights, sample_weight_mode):
    def __init__(self, epochs, verbosity):
        self.epochs = epochs
        self.batch_size = 256
        self.shuffle = True
        self.validation_split = 0.05
        self.optimizer = 'adadelta'
        self.loss = 'mse'
        self.verbosity = verbosity

        self.code_layer_type = None
        self.model = None
        self.sample_weight_mode = None
        self.sample_weights = None
        self.y_true = None
        self.y_pred = None

    def model(self, code_layer_type, input_dim, code_dim):
        self.code_layer_type = code_layer_type
        assert len(code_dim) > 0

        if self.code_layer_type == 'lstm':
            assert len(input_dim) == 2
            input_data = Input(shape=(input_dim[0], input_dim[1]))

            if len(code_dim) == 1:
                encoded = LSTM(code_dim[0])(input_data)
                decoded = RepeatVector(input_dim[0])(encoded)
            elif len(code_dim) > 1:
                encoded = input_data
                for i, units in enumerate(code_dim):
                    if i == len(code_dim) - 1:
                        encoded = LSTM(units)(encoded)
                        continue
                    encoded = LSTM(units, return_sequences=True)(encoded)

                for i, units in enumerate(reversed(code_dim)):
                    if i == 1:
                        decoded = LSTM(units, return_sequences=True)(RepeatVector(input_dim[0])(encoded))
                    elif i > 1:
                        decoded = LSTM(units, return_sequences=True)(decoded)
            else:
                raise ValueError("The codDim must be over 0.")

            decoded = LSTM(input_dim[-1], return_sequences=True)(decoded)
            self.model = Model(input_data, decoded)

        elif self.code_layer_type == 'dense':
            assert len(input_dim) == 1
            input_data = Input(shape=(input_dim[0],))
            encoded = input_data
            for i, units in enumerate(code_dim):
                encoded = Dense(units, activation='relu')(encoded)
            decoded = Dense(input_dim[-1], activation='sigmoid')(encoded)
            self.model = Model(input_data, decoded)

        elif self.code_layer_type == 'cov':
            pass

    def modelMasking(self, code_layer_type, input_dim, code_dim):

        self.code_layer_type = code_layer_type
        assert len(code_dim) > 0

        if self.code_layer_type == 'lstm':
            assert len(input_dim) == 2
            input_data = Input(shape=(input_dim[0], input_dim[1]))
            mask = Masking(mask_value=0.)(input_data)
            if len(code_dim) == 1:
                encoded = LSTM(code_dim[0])(mask)
                decoded = RepeatVector(input_dim[0])(encoded)
            elif len(code_dim) > 1:
                encoded = mask
                for i, units in enumerate(code_dim):
                    if i == len(code_dim) - 1:
                        encoded = LSTM(units)(encoded)
                        continue
                    encoded = LSTM(units, return_sequences=True)(encoded)

                for i, units in enumerate(reversed(code_dim)):
                    if i == 1:
                        decoded = LSTM(units, return_sequences=True)(RepeatVector(input_dim[0])(encoded))
                    elif i > 1:
                        decoded = LSTM(units, return_sequences=True)(decoded)
            else:
                raise ValueError("The codDim must be over 0.")

            decoded = LSTM(input_dim[-1], return_sequences=True)(decoded)
            self.model = Model(input_data, decoded)

        elif self.code_layer_type == 'cov':
            pass
        elif self.code_layer_type == 'dense':
            assert len(input_dim) == 1
            input_data = Input(shape=(input_dim[0],))
            # encoded = input_data
            # for i, units in enumerate(codeDim):
            # 	encoded = Dense(units, activation='relu')(encoded)
            # decoded = Dense(inputDim[-1], activation='sigmoid')(encoded)
            # self.model = Model(input_data, decoded)
            encoder = Dense(code_dim[0], activation="tanh",
                            activity_regularizer=regularizers.l1(10e-5))(input_data)
            encoder = Dense(int(code_dim[0] / 2), activation="relu")(encoder)
            decoder = Dense(int(code_dim[0] / 2), activation='tanh')(encoder)
            decoder = Dense(input_dim[0], activation='relu')(decoder)
            self.model = Model(input_data, decoder)

    def compile(self, *args):

        if len(args) == 0:
            self.model.compile(optimizer=self.optimizer, loss=self.loss)
        elif len(args) == 1:
            if args[0] == 'temporal':
                self.sample_weight_mode = args[0]
                self.model.compile(optimizer=self.optimizer, loss=self.loss, sample_weight_mode=self.sample_weight_mode)
            elif args[0] == 'customFunction':
                self.model.compile(optimizer=self.optimizer, loss=self.weighted_vector_mse)
            else:
                raise ValueError("Invalid maskType, please input 'sample_weights' or 'customFunction'")
        else:
            raise ValueError("argument # must be 0 or 1.")

    def fit(self, *args):

        # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, verbose=1, mode='auto')
        if len(args) == 2:
            if args[1] == 'nor':
                self.model.fit(args[0],
                               args[0],
                               epochs=self.epochs,
                               batch_size=self.batch_size,
                               shuffle=self.shuffle,
                               validation_split=self.validation_split,
                               verbose=self.verbosity)
            # callbacks = [early_stopping])
            elif args[1] == 'rev':
                self.model.fit(args[0],
                               np.flip(args[0], 1),
                               epochs=self.epochs,
                               batch_size=self.batch_size,
                               shuffle=self.shuffle,
                               validation_split=self.validation_split,
                               verbose=self.verbosity)
            # callbacks=[early_stopping])
            else:
                raise ValueError("decoding sequence type: 'normal' or 'reverse'.")

        elif len(args) == 3:
            self.sample_weights = args[2]
            if args[1] == 'nor':
                self.model.fit(args[0],
                               args[0],
                               epochs=self.epochs,
                               batch_size=self.batch_size,
                               shuffle=self.shuffle,
                               validation_split=self.validation_split,
                               sample_weight=self.sample_weights,
                               verbose=self.verbosity)
            # callbacks=[early_stopping])
            elif args[1] == 'rev':
                self.model.fit(args[0],
                               np.flip(args[0], 1),
                               epochs=self.epochs,
                               batch_size=self.batch_size,
                               shuffle=self.shuffle,
                               validation_split=self.validation_split,
                               sample_weight=self.sample_weights,
                               verbose=self.verbosity)
            # callbacks=[early_stopping])
            else:
                raise ValueError("Please input, 'data', 'nor' or 'rev', 'sample_weights'")

    def predict(self, data):
        return self.model.predict(data)

    def weighted_vector_mse(self, y_true, y_pred):

        self.y_true = y_true
        self.y_pred = y_pred

        weight = T.ceil(self.y_true)
        loss = T.square(weight * (self.y_true - self.y_pred))
        # use appropriate relations for other objectives. E.g, for binary_crossentropy:
        # loss = weights * (y_true * T.log(y_pred) + (1.0 - y_true) * T.log(1.0 - y_pred))
        return T.mean(T.sum(loss, axis=-1))
