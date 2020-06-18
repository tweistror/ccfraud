import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from utils.preprocessing.utils import perform_scaling, inverse_scaling, drop_columns, read_csv


class Preprocess_nslkdd:
    def __init__(self, path):
        self.path = path

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
        initial_columns = ["duration", "protocol_type", "service", "flag", "src_bytes",
                           "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                           "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                           "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                           "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                           "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                           "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                           "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                           "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                           "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"]

        train_data = read_csv(self.path['one'], columns=initial_columns)
        test_data = read_csv(self.path['two'], columns=initial_columns)

        data = train_data.append(test_data)
        del train_data, test_data

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
