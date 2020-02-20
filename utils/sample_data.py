import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.list_operations import sample_shuffle, clean_inf_nan


def sample_paysim(x_ben, x_fraud, usv_train, sv_train, sv_train_fraud, test_fraud):
    sv_train_ben = sv_train - sv_train_fraud

    x_usv_train, x_sv_train, y_sv_train, x_test, y_test = \
        data_sampling(x_ben, x_fraud, usv_train, sv_train_ben, sv_train_fraud, test_fraud)

    pca = PCA(n_components=x_usv_train.shape[1])
    x_sv_train = pca.fit_transform(X=x_sv_train)
    x_usv_train = pca.transform(X=x_usv_train)
    x_test = pca.transform(X=x_test)

    sc = StandardScaler()
    x_sv_train = sc.fit_transform(x_sv_train)
    x_usv_train = sc.transform(x_usv_train)
    x_test = sc.transform(x_test)

    return x_usv_train, x_sv_train, y_sv_train, x_test, y_test


def sample_ccfraud(x_ben, x_fraud, usv_train, sv_train, sv_train_fraud, test_fraud):
    sv_train_ben = sv_train - sv_train_fraud

    x_usv_train, x_sv_train, y_sv_train, x_test, y_test = \
        data_sampling(x_ben, x_fraud, usv_train, sv_train_ben, sv_train_fraud, test_fraud)

    # TODO: MinMaxScaler provides strong results for one-class gan (75% f1), but is slightly worse for many usv/sv
    # TODO: baselines - StandardScaler provides poor results for ocgan and slightly better results for many usv/sv
    sc = MinMaxScaler()
    x_sv_train = sc.fit_transform(x_sv_train)
    x_usv_train = sc.transform(x_usv_train)
    x_test = sc.transform(x_test)

    return x_usv_train, x_sv_train, y_sv_train, x_test, y_test


def sample_ieee(x_ben, x_fraud, usv_train, sv_train, sv_train_fraud, test_fraud):
    sv_train_ben = sv_train - sv_train_fraud

    x_usv_train, x_sv_train, y_sv_train, x_test, y_test = \
        data_sampling(x_ben, x_fraud, usv_train, sv_train_ben, sv_train_fraud, test_fraud)

    # Cleaning infinite values to NaN
    x_usv_train = clean_inf_nan(x_usv_train)
    x_sv_train = clean_inf_nan(x_sv_train)
    x_test = clean_inf_nan(x_test)

    return x_usv_train, x_sv_train, y_sv_train, x_test, y_test


def data_sampling(x_ben, x_fraud, usv_train, sv_train_ben, sv_train_fraud, test_fraud):
    # Take random sample of sufficient space (including some offest)
    x_ben = x_ben.sample(n=usv_train + sv_train_ben + sv_train_fraud + test_fraud).values
    x_fraud = x_fraud.sample(n=sv_train_fraud + test_fraud).values

    x_usv_train = x_ben[0:usv_train]
    x_sv_train_ben = x_ben[0:sv_train_ben]
    x_sv_train_fraud = x_fraud[0:sv_train_fraud]
    x_y_sv_train_ben = np.append(x_sv_train_ben, np.zeros((sv_train_ben, 1)), axis=1)
    x_y_sv_train_fraud = np.append(x_sv_train_fraud, np.ones((sv_train_fraud, 1)), axis=1)
    x_y_sv_train = np.concatenate((x_y_sv_train_ben, x_y_sv_train_fraud))
    x_y_sv_train = sample_shuffle(x_y_sv_train)
    x_sv_train = x_y_sv_train[:, :-1]
    y_sv_train = x_y_sv_train[:, -1]

    x_test = x_ben[-test_fraud:].tolist() + x_fraud[-test_fraud:].tolist()
    x_test = np.array(x_test)

    y_test = np.zeros(2 * test_fraud)
    y_test[test_fraud:] = 1

    return x_usv_train, x_sv_train, y_sv_train, x_test, y_test
