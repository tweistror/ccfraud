import pandas as pd
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler

root_path = './data/'


def get_data_ccfraud(path):
    data = load_data(path)

    x_fraud = data.loc[data['Class'] == 1].sample(frac=1)
    x_ben = data.loc[data['Class'] == 0].sample(n=2000)

    scaler = MinMaxScaler()
    x_fraud['scaled_amount'] = scaler.fit_transform(x_fraud['Amount'].values.reshape(-1, 1))
    x_ben['scaled_amount'] = scaler.fit_transform(x_ben['Amount'].values.reshape(-1, 1))

    x_fraud['scaled_time'] = scaler.fit_transform(x_fraud['Time'].values.reshape(-1, 1))
    x_ben['scaled_time'] = scaler.fit_transform(x_ben['Time'].values.reshape(-1, 1))

    x_fraud.drop(['Time', 'Amount', 'Class'], axis=1, inplace=True)
    x_ben.drop(['Time', 'Amount', 'Class'], axis=1, inplace=True)

    x_ben = scaler.fit_transform(x_ben)
    x_fraud = scaler.transform(x_fraud)

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
