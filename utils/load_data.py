from datetime import datetime

import pandas as pd

from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler


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


def get_data_ieee(path, positive_samples=10000, verbosity=0):
    data = load_data(path, verbosity)
    return None


def load_data(path, verbosity=0):
    start_time = datetime.now()
    if verbosity > 0:
        print(f'{path}: Start loading')

    data = pd.read_csv(f'{root_path}{path}')

    time_required = str(datetime.now() - start_time)
    print(f'{path}: Dataset loaded in {time_required}')

    return data
