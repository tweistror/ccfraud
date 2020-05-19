from sklearn.preprocessing import MinMaxScaler

from utils.preprocessing.utils import perform_scaling, inverse_scaling, drop_columns


class Preprocess_ccfraud:
    def __init__(self):
        self.columns = None

        self.scaler = None

        self.columns_to_drop = ['Time', 'Amount']

    def set_columns(self, df):
        self.columns = list(df.columns)

    def preprocess(self, x_sv_train, x_usv_train, x_test):
        scaler, x_sv_train, x_usv_train, x_test = perform_scaling(MinMaxScaler(), x_sv_train, x_usv_train, x_test)
        self.scaler = scaler

        return x_sv_train, x_usv_train, x_test

    def inverse_preprocessing(self, data):
        data = inverse_scaling(self.scaler, data)

        return data

    def initial_processing(self, data):
        # Drop `Time` and `Amount`
        data = drop_columns(data, self.columns_to_drop)

        # Extract fraud and benign transactions and randomize order
        x_fraud = data.loc[data['Class'] == 1]
        x_ben = data.loc[data['Class'] == 0]

        x_fraud = drop_columns(x_fraud, ['Class'])
        x_ben = drop_columns(x_ben, ['Class'])

        return x_ben, x_fraud
