from sklearn.preprocessing import MinMaxScaler

from utils.preprocessing.utils import perform_scaling, inverse_scaling


class Preprocess_ccfraud:
    def __init__(self):
        self.columns = None

        self.scaler = None

    def set_columns(self, df):
        self.columns = list(df.columns)

    def preprocess(self, x_sv_train, x_usv_train, x_test):
        scaler, x_sv_train, x_usv_train, x_test = perform_scaling(MinMaxScaler(), x_sv_train, x_usv_train, x_test)
        self.scaler = scaler

        return x_sv_train, x_usv_train, x_test

    def inverse_preprocessing(self, data):
        data = inverse_scaling(self.scaler, data)

        return data

