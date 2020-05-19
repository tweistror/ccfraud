import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

from utils.list_operations import clean_inf_nan
from utils.preprocessing.utils import perform_scaling, inverse_scaling, perform_pca, inverse_pca, drop_columns, \
    inverse_one_hot_encoding, round_one_hot_endoced_columns


class Preprocess_ieee:
    def __init__(self):
        self.columns = None

        self.scaler = None
        self.pca = None
        self.cat_column_encoders = {}

        self.cat_cols = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2',
                            'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
                            'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21',
                            'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30',
                            'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType',
                         'DeviceInfo']

    def set_columns(self, df):
        self.columns = list(df.columns)

    def add_cat_column_encoder(self, cat_name, encoder):
        self.cat_column_encoders[cat_name] = encoder

    def preprocess(self, x_sv_train, x_usv_train, x_test):
        x_usv_train = clean_inf_nan(x_usv_train)
        x_sv_train = clean_inf_nan(x_sv_train)
        x_test = clean_inf_nan(x_test)

        # pca, x_sv_train, x_usv_train, x_test = perform_pca(x_sv_train, x_usv_train, x_test)
        # self.pca = pca

        scaler, x_sv_train, x_usv_train, x_test = perform_scaling(StandardScaler(), x_sv_train, x_usv_train, x_test)
        self.scaler = scaler

        return x_sv_train, x_usv_train, x_test

    def inverse_preprocessing(self, data):
        data = inverse_scaling(self.scaler, data)
        # data = inverse_pca(self.pca, data)

        df = pd.DataFrame(data=data, columns=self.columns)

        for cat_col in self.cat_column_encoders:
            df[cat_col] = df[cat_col].astype(int)
            df[cat_col] = self.cat_column_encoders[cat_col].inverse_transform(df[cat_col])

        return df

    def initial_processing(self, data):
        # Remove columns with: Only 1 value, many null values and big top values
        # one_value_cols = [col for col in data.columns if data[col].nunique() <= 1]
        # many_null_cols = [col for col in data.columns if data[col].isnull().sum() / data.shape[0] > 0.9]
        # big_top_value_cols = [col for col in data.columns if
        #                       data[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
        # cols_to_drop = list(set(many_null_cols + big_top_value_cols + one_value_cols))
        # cols_to_drop.remove('isFraud')
        # data = data.drop(cols_to_drop, axis=1)

        # Remove dropped cols from cat_cols
        # for i in cols_to_drop:
        #     try:
        #         cat_cols.remove(i)
        #     except ValueError:
        #         pass

        # Label-Encode categorical values
        for col in self.cat_cols:
            if col in data.columns:
                le = LabelEncoder()
                le.fit(list(data[col].astype(str).values))
                data[col] = le.transform(list(data[col].astype(str).values))
                self.add_cat_column_encoder(col, le)

        data = drop_columns(data, ['TransactionDT', 'TransactionID'])

        # Extract `positive_samples` of benign transactions and all fraud transactions
        x_ben = data.loc[data['isFraud'] == 0]
        x_fraud = data.loc[data['isFraud'] == 1]

        x_ben = drop_columns(x_ben, ['isFraud'])
        x_fraud = drop_columns(x_fraud, ['isFraud'])

        return x_ben, x_fraud
