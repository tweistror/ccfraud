import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from utils.preprocessing.utils import perform_scaling, inverse_scaling, perform_pca, inverse_pca, drop_columns, \
    inverse_one_hot_encoding, round_one_hot_endoced_columns, one_hot_encode_column


class Preprocess_paysim_custom:
    def __init__(self):
        self.columns = None

        self.scaler = None
        self.pca = None

        self.columns_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud']

    def set_columns(self, df):
        self.columns = list(df.columns)

    def preprocess(self, x_sv_train, x_usv_train, x_test):
        pca, x_sv_train, x_usv_train, x_test = perform_pca(x_sv_train, x_usv_train, x_test)
        self.pca = pca

        scaler, x_sv_train, x_usv_train, x_test = perform_scaling(MinMaxScaler(), x_sv_train, x_usv_train, x_test)
        self.scaler = scaler

        return x_sv_train, x_usv_train, x_test

    def inverse_preprocessing(self, data):
        data = inverse_scaling(self.scaler, data)
        data = inverse_pca(self.pca, data)

        data = round_one_hot_endoced_columns(data, -5)

        df = pd.DataFrame(data=data, columns=self.columns)

        df = inverse_one_hot_encoding(df, 'action', -5)

        return df

    def initial_processing(self, data):
        # Add feature for `nameOrig` to `nameDest` relation with one-hot encoding
        # => Feature is not important
        # data['nameOrig'] = data['nameOrig'].apply(lambda x: x[:1])
        # data['nameDest'] = data['nameDest'].apply(lambda x: x[:1])
        # data['from_to'] = data['nameOrig'] + data['nameDest']
        # data = pd.concat([data, pd.get_dummies(data['from_to'], prefix='from_to')], axis=1)
        # data.drop(columns=['from_to'], inplace=True)

        data = drop_columns(data, self.columns_to_drop)
        data = one_hot_encode_column(data, 'action')

        # Extract fraud and benign transactions and randomize order
        x_fraud = data.loc[data['isFraud'] == 1]
        x_ben = data.loc[data['isFraud'] == 0]

        x_fraud = drop_columns(x_fraud, ['isFraud'])
        x_ben = drop_columns(x_ben, ['isFraud'])

        return x_ben, x_fraud
