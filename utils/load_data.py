import pandas as pd

from sklearn.preprocessing import RobustScaler, MinMaxScaler


root_path = './data/'


def get_data_paysim(path, verbose=0):
    # TODO
    return None


def get_data_ccfraud(path, verbose=0):
    data = pd.read_csv(f'{root_path}{path}')

    scaler = MinMaxScaler()
    # scaler = RobustScaler()

    x = data.loc[:, data.columns != 'Class']
    y = data.loc[:, 'Class']

    x = scaler.fit_transform(x.values)
    y = y.values

    return x[y == 0], x[y == 1]


def get_data_ieee(path, verbose=0):
    # TODO
    return None
