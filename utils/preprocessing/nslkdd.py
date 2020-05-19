import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils.preprocessing.utils import perform_scaling, inverse_scaling


class Preprocess_nslkdd:
    def __init__(self):
        self.columns = None

        self.scaler = None
        self.pca = None
        self.train_test_dimensions = None
        self.scale_number = 255
        self.cat_column_encoders = {}

    def set_columns(self, df):
        self.columns = list(df.columns)

    def add_cat_column_encoder(self, cat_name, encoder):
        self.cat_column_encoders[cat_name] = encoder

    def set_train_test_dimensions(self, train_test_dimensions):
        self.train_test_dimensions = train_test_dimensions

    def preprocess(self, x_sv_train, x_usv_train, x_test):
        scaler, x_sv_train, x_usv_train, x_test = perform_scaling(MinMaxScaler(), x_sv_train, x_usv_train, x_test)
        self.scaler = scaler

        return x_sv_train, x_usv_train, x_test

    def inverse_preprocessing(self, data):
        data = inverse_scaling(self.scaler, data)

        df = pd.DataFrame(data=data, columns=self.columns)

        for cat_col in self.cat_column_encoders:
            df[cat_col] = df[cat_col].astype(int)
            df[cat_col] = self.cat_column_encoders[cat_col].inverse_transform(df[cat_col])

        return df
