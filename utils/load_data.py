from datetime import datetime

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from utils.data_loading.mnist import get_data_mnist
from utils.data_loading.saperp_synthetic import get_data_saperp


class LoadData(object):
    def __init__(self, dataset_string, path, seed, parameter_class, verbosity=0):
        self.dataset_string = dataset_string
        self.verbosity = verbosity
        self.path = path
        self.seed = seed
        self.parameter_class = parameter_class

    def get_data(self):
        if self.dataset_string == "paysim":
            x_ben, x_fraud = self.get_data_paysim()
        if self.dataset_string == "paysim-custom":
            x_ben, x_fraud = self.get_data_paysim_custom()
        elif self.dataset_string == "ccfraud":
            x_ben, x_fraud = self.get_data_ccfraud()
        elif self.dataset_string == "ieee":
            x_ben, x_fraud = self.get_data_ieee()
        elif self.dataset_string == "nslkdd":
            x_ben, x_fraud = self.get_data_nslkdd()
        elif self.dataset_string == "saperp-ek" or self.dataset_string == "saperp-vk":
            fraud_only = self.parameter_class.get_saperp_mode()['fraud_only']
            x_ben, x_fraud = get_data_saperp(self.dataset_string, self.path, self.seed, fraud_only)
        elif self.dataset_string == "mnist":
            fraud_number = self.parameter_class.get_mnist_mode()['fraud_number']
            x_ben, x_fraud = get_data_mnist(self.path, self.seed, fraud_number)

        return x_ben, x_fraud

    def get_data_paysim(self):
        data = self.read_csv(self.path['one'])

        # Add feature for `nameOrig` to `nameDest` relation with one-hot encoding
        #  => Feature is not important
        # data['nameOrig'] = data['nameOrig'].apply(lambda x: x[:1])
        # data['nameDest'] = data['nameDest'].apply(lambda x: x[:1])
        # data['from_to'] = data['nameOrig'] + data['nameDest']
        # data = pd.concat([data, pd.get_dummies(data['from_to'], prefix='from_to')], axis=1)
        # data.drop(columns=['from_to'], inplace=True)

        # Drop `isFlaggedFraud` column
        data.drop(columns=['nameOrig', 'nameDest', 'isFlaggedFraud'], inplace=True)

        # One-hot encode type
        data = pd.concat([data, pd.get_dummies(data['type'], prefix='type')], axis=1)
        data.drop(['type'], axis=1, inplace=True)

        # Extract fraud and benign transactions and randomize order
        x_fraud = data.loc[data['isFraud'] == 1].sample(frac=1, random_state=self.seed)
        x_ben = data.loc[data['isFraud'] == 0].sample(frac=1, random_state=self.seed)

        x_fraud.drop(['isFraud'], axis=1, inplace=True)
        x_ben.drop(['isFraud'], axis=1, inplace=True)

        return x_ben, x_fraud

    def get_data_ccfraud(self):
        data = self.read_csv(self.path['one'])

        # Drop `Time` and `Amount`
        data.drop(['Time', 'Amount'], axis=1, inplace=True)

        # Extract fraud and benign transactions and randomize order
        x_fraud = data.loc[data['Class'] == 1].sample(frac=1, random_state=self.seed)
        x_ben = data.loc[data['Class'] == 0].sample(frac=1, random_state=self.seed)

        x_fraud.drop(['Class'], axis=1, inplace=True)
        x_ben.drop(['Class'], axis=1, inplace=True)

        return x_ben, x_fraud

    def get_data_paysim_custom(self):
        data = self.read_csv(self.path['one'])

        # Add feature for `nameOrig` to `nameDest` relation with one-hot encoding
        # => Feature is not important
        # data['nameOrig'] = data['nameOrig'].apply(lambda x: x[:1])
        # data['nameDest'] = data['nameDest'].apply(lambda x: x[:1])
        # data['from_to'] = data['nameOrig'] + data['nameDest']
        # data = pd.concat([data, pd.get_dummies(data['from_to'], prefix='from_to')], axis=1)
        # data.drop(columns=['from_to'], inplace=True)

        # Drop `isFlaggedFraud` column
        data.drop(columns=['nameOrig', 'nameDest', 'isFlaggedFraud'], inplace=True)

        # One-hot encode type
        data = pd.concat([data, pd.get_dummies(data['action'], prefix='action')], axis=1)
        data.drop(['action'], axis=1, inplace=True)

        # Extract fraud and benign transactions and randomize order
        x_fraud = data.loc[data['isFraud'] == 1].sample(frac=1, random_state=self.seed)
        x_ben = data.loc[data['isFraud'] == 0].sample(frac=1, random_state=self.seed)

        x_fraud.drop(['isFraud'], axis=1, inplace=True)
        x_ben.drop(['isFraud'], axis=1, inplace=True)

        return x_ben, x_fraud

    def get_data_ieee(self):
        # if skip is True:
        #     start_time = datetime.now()
        #     x_ben = pd.read_csv('./debug/ieee/x_ben.csv')
        #     x_fraud = pd.read_csv('./debug/ieee/x_fraud.csv')
        #     if verbosity > 0:
        #         print(f'IEEE: Preprocessed dataset loaded in {str(datetime.now() - start_time)}')
        #
        #     return x_ben, x_fraud

        transaction_data = self.read_csv(self.path['one'])
        identity_data = self.read_csv(self.path['two'])

        data = pd.merge(transaction_data, identity_data, on='TransactionID', how='left')
        del transaction_data, identity_data

        # Remove columns with: Only 1 value, many null values and big top values
        # one_value_cols = [col for col in data.columns if data[col].nunique() <= 1]
        # many_null_cols = [col for col in data.columns if data[col].isnull().sum() / data.shape[0] > 0.9]
        # big_top_value_cols = [col for col in data.columns if
        #                       data[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
        # cols_to_drop = list(set(many_null_cols + big_top_value_cols + one_value_cols))
        # cols_to_drop.remove('isFraud')
        # data = data.drop(cols_to_drop, axis=1)

        cat_cols = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2',
                    'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
                    'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21',
                    'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30',
                    'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']

        # Remove dropped cols from cat_cols
        # for i in cols_to_drop:
        #     try:
        #         cat_cols.remove(i)
        #     except ValueError:
        #         pass

        # Label-Encode categorical values
        for col in cat_cols:
            if col in data.columns:
                le = LabelEncoder()
                le.fit(list(data[col].astype(str).values))
                data[col] = le.transform(list(data[col].astype(str).values))

        data.drop(['TransactionDT', 'TransactionID'], axis=1, inplace=True)

        # Extract `positive_samples` of benign transactions and all fraud transactions
        x_ben = data.loc[data['isFraud'] == 0].sample(frac=1, random_state=self.seed)
        x_fraud = data.loc[data['isFraud'] == 1].sample(frac=1, random_state=self.seed)

        x_fraud.drop(['isFraud'], axis=1, inplace=True)
        x_ben.drop(['isFraud'], axis=1, inplace=True)

        # x_ben.to_csv(r'x_ben.csv')
        # x_fraud.to_csv(r'x_fraud.csv')

        return x_ben, x_fraud

    def get_data_nslkdd(self):
        columns = ["duration", "protocol_type", "service", "flag", "src_bytes",
                   "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                   "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                   "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                   "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                   "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                   "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                   "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                   "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                   "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"]

        train_data = self.read_csv(self.path['one'], columns=columns)
        test_data = self.read_csv(self.path['two'], columns=columns)

        attack_mapping = {
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

        # Map various attacks to predefined labels
        train_data['label'] = train_data['label'].map(attack_mapping)
        test_data['label'] = test_data['label'].map(attack_mapping)

        data = train_data.append(test_data)
        del train_data, test_data

        data.drop(['difficulty'], axis=1, inplace=True)

        cat_cols = ['protocol_type', 'service', 'flag']

        for col in cat_cols:
            if col in data.columns:
                le = LabelEncoder()
                le.fit(list(data[col].astype(str).values))
                data[col] = le.transform(list(data[col].astype(str).values))

        x_ben = data.loc[data['label'] == 'normal'].sample(frac=1, random_state=self.seed)
        x_fraud = data.loc[data['label'] != 'normal'].sample(frac=1, random_state=self.seed)

        x_ben.drop(['label'], axis=1, inplace=True)
        x_fraud.drop(['label'], axis=1, inplace=True)

        return x_ben, x_fraud

    def read_csv(self, path, columns=None):
        start_time = datetime.now()
        if self.verbosity > 0:
            print(f'{path}: Start loading dataset')

        data = pd.read_csv(f'{path}', names=columns)

        if self.verbosity > 0:
            time_required = str(datetime.now() - start_time)
            print(f'{path}: Dataset loaded in {time_required}')

        return data


# Copy in desired `get`-method for heatmap of feature corr
# # Using Pearson Correlation
# plt.figure(figsize=(12, 10))
# cor = data.corr()
# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds, mask=cor < 0.1)
# plt.show()
# exit(0)
