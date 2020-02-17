from datetime import datetime

import pandas as pd

from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, LabelEncoder

from utils.list_operations import clean_inf_nan

root_path = './data/'


def get_data_paysim(path, positive_samples=10000, verbosity=0):
    data = load_data(path, verbosity)

    # Drop unnecessary columns
    # TODO: Add feature for nameOrig/nameDest relation
    data = data.drop(columns=['nameOrig', 'nameDest', 'isFlaggedFraud'])

    # Extract `positive_samples` of benign transactions and all fraud transactions
    benign = data.loc[data['isFraud'] == 0].sample(positive_samples)
    fraud = data.loc[data['isFraud'] == 1].sample(frac=1)
    extracted_data = pd.concat([benign, fraud])

    # One-hot encode `type`
    extracted_data = pd.concat([extracted_data, pd.get_dummies(extracted_data['type'], prefix='type')], axis=1)
    extracted_data = extracted_data.drop(['type'], axis=1)

    x = extracted_data.loc[:, extracted_data.columns != 'isFraud']
    y = extracted_data.loc[:, 'isFraud']

    scaler = StandardScaler()
    x = scaler.fit_transform(x.values)
    y = y.values

    return x[y == 0], x[y == 1]


def get_data_ccfraud(path, positive_samples=10000, verbosity=0):
    data = load_data(path, verbosity)

    # Extract `positive_samples` of benign transactions and all fraud transactions
    benign = data.loc[data['Class'] == 0].sample(positive_samples)
    fraud = data.loc[data['Class'] == 1].sample(frac=1)
    extracted_data = pd.concat([benign, fraud])

    scaler = MinMaxScaler()
    # scaler = RobustScaler()

    x = extracted_data.loc[:, extracted_data.columns != 'Class']
    y = extracted_data.loc[:, 'Class']

    x = scaler.fit_transform(x.values)
    y = y.values

    return x[y == 0], x[y == 1]


def get_data_ieee(transaction_path, identity_path, positive_samples=10000, verbosity=0):
    transaction_data = load_data(transaction_path, verbosity)
    identity_data = load_data(identity_path, verbosity)

    data = pd.merge(transaction_data, identity_data, on='TransactionID', how='left')
    del transaction_data, identity_data

    # Remove columns with: Only 1 value, many null values and big top values
    one_value_cols = [col for col in data.columns if data[col].nunique() <= 1]
    many_null_cols = [col for col in data.columns if data[col].isnull().sum() / data.shape[0] > 0.9]
    big_top_value_cols = [col for col in data.columns if
                          data[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
    cols_to_drop = list(set(many_null_cols + big_top_value_cols + one_value_cols))
    cols_to_drop.remove('isFraud')
    data = data.drop(cols_to_drop, axis=1)

    cat_cols = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2',
                'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
                'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21',
                'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30',
                'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']

    # Remove dropped cols from cat_cols
    for i in cols_to_drop:
        try:
            cat_cols.remove(i)
        except ValueError:
            pass

    # Extract `positive_samples` of benign transactions and all fraud transactions
    benign = data.loc[data['isFraud'] == 0].sample(positive_samples)
    fraud = data.loc[data['isFraud'] == 1].sample(frac=1)
    data = pd.concat([benign, fraud])

    # Label-Encode categorical values
    for col in cat_cols:
        if col in data.columns:
            le = LabelEncoder()
            le.fit(list(data[col].astype(str).values))
            data[col] = le.transform(list(data[col].astype(str).values))

    data = data.drop(['TransactionDT', 'TransactionID'], axis=1)

    x = data.loc[:, data.columns != 'isFraud']
    y = data.loc[:, 'isFraud']

    x = x.values
    y = y.values

    # Cleaning infinite values to NaN
    x = clean_inf_nan(x)

    return x[y == 0], x[y == 1]


def load_data(path, verbosity=0):
    start_time = datetime.now()
    if verbosity > 0:
        print(f'{path}: Start loading dataset')

    data = pd.read_csv(f'{root_path}{path}')

    if verbosity > 0:
        time_required = str(datetime.now() - start_time)
        print(f'{path}: Dataset loaded in {time_required}')

    return data


def get_parameters(dataset_string):
    if dataset_string == 'paysim':
        usv_train = 5000
        sv_train = 2000
        sv_train_fraud = 50
        test_fraud = 1000
    elif dataset_string == 'ccfraud':
        usv_train = 700
        sv_train = 1000
        sv_train_fraud = 10
        test_fraud = 490
    elif dataset_string == 'ieee':
        usv_train = 1000
        sv_train = 2000
        sv_train_fraud = 50
        test_fraud = 500

    return usv_train, sv_train, sv_train_fraud, test_fraud
