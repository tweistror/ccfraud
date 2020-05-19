import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from utils.preprocessing.utils import perform_scaling, inverse_scaling, drop_columns


class Preprocess_nslkdd:
    def __init__(self):
        self.columns = None

        self.scaler = None
        self.pca = None
        self.train_test_dimensions = None
        self.scale_number = 255
        self.cat_column_encoders = {}

        self.columns_to_drop = ['difficulty']
        self.cat_cols = ['protocol_type', 'service', 'flag']

        self.attack_mapping = {
            'normal': 'normal',

            'back': 'dos',
            'land': 'dos',
            'neptune': 'dos',
            'pod': 'dos',
            'smurf': 'dos',
            'teardrop': 'dos',
            'apache2': 'dos',
            'udpstorm': 'dos',
            'processtable': 'dos',
            'worm': 'dos',

            'satan': 'probe',
            'ipsweep': 'probe',
            'nmap': 'probe',
            'portsweep': 'probe',
            'mscan': 'probe',
            'saint': 'probe',

            'guess_passwd': 'R2L',
            'ftp_write': 'R2L',
            'imap': 'R2L',
            'phf': 'R2L',
            'multihop': 'R2L',
            'warezmaster': 'R2L',
            'warezclient': 'R2L',
            'spy': 'R2L',
            'xlock': 'R2L',
            'xsnoop': 'R2L',
            'snmpguess': 'R2L',
            'snmpgetattack': 'R2L',
            'httptunnel': 'R2L',
            'sendmail': 'R2L',
            'named': 'R2L',

            'buffer_overflow': 'U2R',
            'loadmodule': 'U2R',
            'rootkit': 'U2R',
            'perl': 'U2R',
            'sqlattack': 'U2R',
            'xterm': 'U2R',
            'ps': 'U2R'
        }

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

    def initial_processing(self, data):
        data['label'] = data['label'].map(self.attack_mapping)

        data = drop_columns(data, self.columns_to_drop)

        for col in self.cat_cols:
            if col in data.columns:
                le = LabelEncoder()
                le.fit(list(data[col].astype(str).values))
                data[col] = le.transform(list(data[col].astype(str).values))
                self.add_cat_column_encoder(col, le)

        x_ben = data.loc[data['label'] == 'normal']
        x_fraud = data.loc[data['label'] != 'normal']

        x_ben = drop_columns(x_ben, ['label'])
        x_fraud = drop_columns(x_fraud, ['label'])

        return x_ben, x_fraud
