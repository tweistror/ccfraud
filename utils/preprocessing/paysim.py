import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from utils.preprocessing.utils import perform_scaling, inverse_scaling, perform_pca, inverse_pca, drop_columns, \
    inverse_one_hot_encoding, round_one_hot_endoced_columns


class Preprocess_paysim:
    def __init__(self):
        self.columns = None

        self.scaler = None
        self.pca = None

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
