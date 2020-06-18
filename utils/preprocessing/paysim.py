import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from utils.preprocessing.utils import perform_scaling, inverse_scaling, perform_pca, inverse_pca, drop_columns, \
    inverse_one_hot_encoding, round_one_hot_endoced_columns, one_hot_encode_column, read_csv


class Preprocess_paysim:
    def __init__(self, path):
        self.path = path

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

        df = inverse_one_hot_encoding(df, 'type', -5)

        return df

    def initial_processing(self):
        data = read_csv(self.path['one'])

        data = drop_columns(data, self.columns_to_drop)
        data = one_hot_encode_column(data, 'type')

        # Extract fraud and benign transactions and randomize order
        x_fraud = data.loc[data['isFraud'] == 1]
        x_ben = data.loc[data['isFraud'] == 0]

        x_fraud = drop_columns(x_fraud, ['isFraud'])
        x_ben = drop_columns(x_ben, ['isFraud'])

        return x_ben, x_fraud
