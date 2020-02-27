import numpy as np
import theano.tensor as T

from keras.layers import Input, Dense, LSTM, RepeatVector
from keras.models import Sequential, Model
from keras.layers.core import Masking
from keras import regularizers


class Dense_Autoencoder(object):
    """docstring for LSTM_Autoencoder"""

    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.autoencoder = Autoencoder()
        self.autoencoder.modelMasking('dense', [self.input_dim], self.hidden_dim)

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

    # def __init__(self, sampleWeights, sample_weight_mode):
    def __init__(self):
        # super(Autoencoder, self).__init__()
        # self.codeLayerType = 'dense'
        self.nb_epoch = 20
        self.batch_size = 256
        self.shuffle = True
        self.validation_split = 0.05
        self.optimizer = 'adadelta'
        self.loss = 'mse'

    # self.sampleWeights = sampleWeights
    # self.sample_weight_mode = sample_weight_mode

    def model(self, codeLayerType, inputDim, codeDim):

        self.codeLayerType = codeLayerType
        assert len(codeDim) > 0

        if self.codeLayerType == 'lstm':
            assert len(inputDim) == 2
            inputData = Input(shape=(inputDim[0], inputDim[1]))

            if len(codeDim) == 1:
                encoded = LSTM(codeDim[0])(inputData)
                decoded = RepeatVector(inputDim[0])(encoded)
            elif len(codeDim) > 1:
                encoded = inputData
                for i, units in enumerate(codeDim):
                    if i == len(codeDim) - 1:
                        encoded = LSTM(units)(encoded)
                        continue
                    encoded = LSTM(units, return_sequences=True)(encoded)

                for i, units in enumerate(reversed(codeDim)):
                    if i == 1:
                        decoded = LSTM(units, return_sequences=True)(RepeatVector(inputDim[0])(encoded))
                    elif i > 1:
                        decoded = LSTM(units, return_sequences=True)(decoded)
            else:
                raise ValueError("The codDim must be over 0.")

            decoded = LSTM(inputDim[-1], return_sequences=True)(decoded)
            self.model = Model(inputData, decoded)

        elif self.codeLayerType == 'dense':
            assert len(inputDim) == 1
            inputData = Input(shape=(inputDim[0],))
            encoded = inputData
            for i, units in enumerate(codeDim):
                encoded = Dense(units, activation='relu')(encoded)
            decoded = Dense(inputDim[-1], activation='sigmoid')(encoded)
            self.model = Model(inputData, decoded)

        elif self.codeLayerType == 'cov':
            pass

    def modelMasking(self, codeLayerType, inputDim, codeDim):

        self.codeLayerType = codeLayerType
        assert len(codeDim) > 0

        if self.codeLayerType == 'lstm':
            assert len(inputDim) == 2
            inputData = Input(shape=(inputDim[0], inputDim[1]))
            mask = Masking(mask_value=0.)(inputData)
            if len(codeDim) == 1:
                encoded = LSTM(codeDim[0])(mask)
                decoded = RepeatVector(inputDim[0])(encoded)
            elif len(codeDim) > 1:
                encoded = mask
                for i, units in enumerate(codeDim):
                    if i == len(codeDim) - 1:
                        encoded = LSTM(units)(encoded)
                        continue
                    encoded = LSTM(units, return_sequences=True)(encoded)

                for i, units in enumerate(reversed(codeDim)):
                    if i == 1:
                        decoded = LSTM(units, return_sequences=True)(RepeatVector(inputDim[0])(encoded))
                    elif i > 1:
                        decoded = LSTM(units, return_sequences=True)(decoded)
            else:
                raise ValueError("The codDim must be over 0.")

            decoded = LSTM(inputDim[-1], return_sequences=True)(decoded)
            self.model = Model(inputData, decoded)

        elif self.codeLayerType == 'cov':
            pass
        elif self.codeLayerType == 'dense':
            assert len(inputDim) == 1
            inputData = Input(shape=(inputDim[0],))
            # encoded = inputData
            # for i, units in enumerate(codeDim):
            # 	encoded = Dense(units, activation='relu')(encoded)
            # decoded = Dense(inputDim[-1], activation='sigmoid')(encoded)
            # self.model = Model(inputData, decoded)
            encoder = Dense(codeDim[0], activation="tanh",
                            activity_regularizer=regularizers.l1(10e-5))(inputData)
            encoder = Dense(int(codeDim[0] / 2), activation="relu")(encoder)
            decoder = Dense(int(codeDim[0] / 2), activation='tanh')(encoder)
            decoder = Dense(inputDim[0], activation='relu')(decoder)
            self.model = Model(inputData, decoder)

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
                raise ValueError("Invalid maskType, please input 'sampleWeights' or 'customFunction'")
        else:
            raise ValueError("argument # must be 0 or 1.")

    def fit(self, *args):

        # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, verbose=1, mode='auto')
        if len(args) == 2:
            if args[1] == 'nor':
                self.model.fit(args[0],
                               args[0],
                               nb_epoch=self.nb_epoch,
                               batch_size=self.batch_size,
                               shuffle=self.shuffle,
                               validation_split=self.validation_split)
            # callbacks = [early_stopping])
            elif args[1] == 'rev':
                self.model.fit(args[0],
                               np.flip(args[0], 1),
                               nb_epoch=self.nb_epoch,
                               batch_size=self.batch_size,
                               shuffle=self.shuffle,
                               validation_split=self.validation_split)
            # callbacks=[early_stopping])
            else:
                raise ValueError("decoding sequence type: 'normal' or 'reverse'.")

        elif len(args) == 3:
            self.sampleWeights = args[2]
            if args[1] == 'nor':
                self.model.fit(args[0],
                               args[0],
                               nb_epoch=self.nb_epoch,
                               batch_size=self.batch_size,
                               shuffle=self.shuffle,
                               validation_split=self.validation_split,
                               sample_weight=self.sampleWeights)
            # callbacks=[early_stopping])
            elif args[1] == 'rev':
                self.model.fit(args[0],
                               np.flip(args[0], 1),
                               nb_epoch=self.nb_epoch,
                               batch_size=self.batch_size,
                               shuffle=self.shuffle,
                               validation_split=self.validation_split,
                               sample_weight=self.sampleWeights)
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
