from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

root_path = './data/'

# Copy in desired `get`-method for heatmap of feature corr
# # Using Pearson Correlation
# plt.figure(figsize=(12, 10))
# cor = data.corr()
# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds, mask=cor < 0.1)
# plt.show()
# exit(0)


def get_data_paysim(path, verbosity=0):
    data = load_data(path, verbosity)

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
    x_fraud = data.loc[data['isFraud'] == 1].sample(frac=1)
    x_ben = data.loc[data['isFraud'] == 0].sample(frac=1)

    x_fraud.drop(['isFraud'], axis=1, inplace=True)
    x_ben.drop(['isFraud'], axis=1, inplace=True)

    return x_ben, x_fraud


def get_data_paysim_custom(path, verbosity=0):
    data = load_data(path, verbosity)

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
    x_fraud = data.loc[data['isFraud'] == 1].sample(frac=1)
    x_ben = data.loc[data['isFraud'] == 0].sample(frac=1)

    x_fraud.drop(['isFraud'], axis=1, inplace=True)
    x_ben.drop(['isFraud'], axis=1, inplace=True)

    return x_ben, x_fraud


def get_data_ccfraud(path, verbosity=0):
    data = load_data(path, verbosity)

    # Drop `Time` and `Amount`
    data.drop(['Time', 'Amount'], axis=1, inplace=True)

    # Extract fraud and benign transactions and randomize order
    x_fraud = data.loc[data['Class'] == 1].sample(frac=1)
    x_ben = data.loc[data['Class'] == 0].sample(frac=1)

    x_fraud.drop(['Class'], axis=1, inplace=True)
    x_ben.drop(['Class'], axis=1, inplace=True)

    return x_ben, x_fraud


def get_data_ieee(transaction_path, identity_path, verbosity=0, skip=False):
    # TODO: Do not forget to remove skip
    # if skip is True:
    #     start_time = datetime.now()
    #     x_ben = pd.read_csv('./debug/ieee/x_ben.csv')
    #     x_fraud = pd.read_csv('./debug/ieee/x_fraud.csv')
    #     if verbosity > 0:
    #         print(f'IEEE: Preprocessed dataset loaded in {str(datetime.now() - start_time)}')
    #
    #     return x_ben, x_fraud

    transaction_data = load_data(transaction_path, verbosity)
    identity_data = load_data(identity_path, verbosity)

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
    x_ben = data.loc[data['isFraud'] == 0].sample(frac=1)
    x_fraud = data.loc[data['isFraud'] == 1].sample(frac=1)

    x_fraud.drop(['isFraud'], axis=1, inplace=True)
    x_ben.drop(['isFraud'], axis=1, inplace=True)

    # x_ben.to_csv(r'x_ben.csv')
    # x_fraud.to_csv(r'x_fraud.csv')

    return x_ben, x_fraud


def load_data(path, verbosity=0):
    start_time = datetime.now()
    if verbosity > 0:
        print(f'{path}: Start loading dataset')

    data = pd.read_csv(f'{root_path}{path}')

    if verbosity > 0:
        time_required = str(datetime.now() - start_time)
        print(f'{path}: Dataset loaded in {time_required}')

    return data


def get_parameters(dataset_string, config, cross_validation_count=0):
    factor = 0 if cross_validation_count < 2 else cross_validation_count - 1
    cfg = config['DEFAULT']

    if dataset_string == 'paysim' or dataset_string == 'paysim_custom':
        usv_train = int(cfg['paysim_usv_train'])
        sv_train = int(cfg['paysim_sv_train'])
        sv_train_fraud = int(cfg['paysim_sv_train_fraud'])
        test_benign = 1000 - factor * int(cfg['paysim_sv_train_fraud'])
        test_fraud = 1000 - factor * int(cfg['paysim_sv_train_fraud'])
    elif dataset_string == 'ccfraud':
        usv_train = int(cfg['ccfraud_usv_train'])
        sv_train = int(cfg['ccfraud_sv_train'])
        sv_train_fraud = int(cfg['ccfraud_sv_train_fraud'])
        test_benign = 450 - factor * int(cfg['ccfraud_sv_train_fraud'])
        test_fraud = 450 - factor * int(cfg['ccfraud_sv_train_fraud'])
    elif dataset_string == 'ieee':
        usv_train = int(cfg['ieee_usv_train'])
        sv_train = int(cfg['ieee_sv_train'])
        sv_train_fraud = int(cfg['ieee_sv_train_fraud'])
        test_benign = 5000 - factor * int(cfg['ieee_sv_train_fraud'])
        test_fraud = 5000 - factor * int(cfg['ieee_sv_train_fraud'])

    return usv_train, sv_train, sv_train_fraud, test_fraud, test_benign
