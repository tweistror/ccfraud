import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def drop_columns(df, columns):
    df = df.drop(columns=columns)

    return df


def one_hot_encode_column(df, column_name):
    df = pd.concat([df, pd.get_dummies(df[column_name], prefix=column_name)], axis=1)
    df = df.drop([column_name], axis=1)

    return df


def round_one_hot_endoced_columns(data, index):
    rounded_columns = np.around(data[:, index:], decimals=0).astype(int)
    data = np.append(data[:, :index], rounded_columns, axis=1)

    return data


def inverse_one_hot_encoding(df, column_name, start_index):
    one_hot_df = df.iloc[:, start_index:]

    one_hot_df_column_names = list(one_hot_df.columns)
    one_hot_df = drop_columns(one_hot_df[one_hot_df == 1].stack().reset_index().drop(0, 1), ['level_0'])
    one_hot_df = one_hot_df['level_1'].str.replace(f'{column_name}_', '')
    df = drop_columns(df, columns=one_hot_df_column_names)
    df[column_name] = one_hot_df

    return df


def perform_pca(x_sv_train, x_usv_train, x_test):
    pca = PCA(n_components=x_usv_train.shape[1])

    if len(x_sv_train) > len(x_usv_train):
        x_sv_train = pca.fit_transform(X=x_sv_train)
        x_usv_train = pca.transform(X=x_usv_train)
    else:
        x_usv_train = pca.fit_transform(x_usv_train)
        x_sv_train = pca.transform(x_sv_train)
    x_test = pca.transform(X=x_test)

    return pca, x_sv_train, x_usv_train, x_test


def inverse_pca(pca, data):
    data = pca.inverse_transform(data)

    return data


def perform_scaling(scaler, x_sv_train, x_usv_train, x_test):
    if len(x_sv_train) > len(x_usv_train):
        x_sv_train = scaler.fit_transform(x_sv_train)
        x_usv_train = scaler.transform(x_usv_train)
    else:
        x_usv_train = scaler.fit_transform(x_usv_train)
        x_sv_train = scaler.transform(x_sv_train)
    x_test = scaler.transform(x_test)

    return scaler, x_sv_train, x_usv_train, x_test


def inverse_scaling(scaler, data):
    data = scaler.inverse_transform(data)

    return data


def is_image_dataset(dataset_string):
    if dataset_string == "mnist" or dataset_string == "cifar10":
        return True
    return False


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        _dict = pickle.load(fo, encoding='bytes')
    return _dict
