import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.list_operations import sample_shuffle, clean_inf_nan


def split_and_preprocess_data(dataset_string, x_ben, x_fraud, usv_train, sv_train, sv_train_fraud, test_fraud,
                              test_benign, cross_validation_k):
    if dataset_string == "paysim" or dataset_string == "paysim_custom":
        x_usv_train, x_sv_train, y_sv_train, x_test, y_test = with_paysim(x_ben, x_fraud, usv_train, sv_train,
                                                                          sv_train_fraud, test_fraud,  test_benign,
                                                                          cross_validation_k,
                                                                          is_custom=dataset_string == "paysim_custom")
    elif dataset_string == "ccfraud":
        x_usv_train, x_sv_train, y_sv_train, x_test, y_test = with_ccfraud(x_ben, x_fraud, usv_train, sv_train,
                                                                           sv_train_fraud, test_fraud, test_benign,
                                                                           cross_validation_k)
    elif dataset_string == "ieee":
        x_usv_train, x_sv_train, y_sv_train, x_test, y_test = with_ieee(x_ben, x_fraud, usv_train, sv_train,
                                                                        sv_train_fraud, test_fraud, test_benign,
                                                                        cross_validation_k)

    return x_usv_train, x_sv_train, y_sv_train, x_test, y_test


def with_paysim(x_ben, x_fraud, usv_train, sv_train, sv_train_fraud, test_fraud, test_benign, cross_validation_k,
                is_custom):
    sv_train_ben = sv_train - sv_train_fraud

    x_usv_train, x_sv_train, y_sv_train, x_test, y_test = \
        data_sampling(x_ben, x_fraud, usv_train, sv_train_ben, sv_train_fraud, test_fraud, test_benign,
                      cross_validation_k)

    # TODO: Leave PCA out completely or not
    # if is_custom is False:
    #     pca = PCA(n_components=x_usv_train.shape[1])
    #     if len(x_sv_train) > len(x_usv_train):
    #         x_sv_train = pca.fit_transform(X=x_sv_train)
    #         x_usv_train = pca.transform(X=x_usv_train)
    #     else:
    #         x_usv_train = pca.fit_transform(x_usv_train)
    #         x_sv_train = pca.transform(x_sv_train)
    #     x_test = pca.transform(X=x_test)

    sc = StandardScaler()
    if len(x_sv_train) > len(x_usv_train):
        x_sv_train = sc.fit_transform(x_sv_train)
        x_usv_train = sc.transform(x_usv_train)
    else:
        x_usv_train = sc.fit_transform(x_usv_train)
        x_sv_train = sc.transform(x_sv_train)
    x_test = sc.transform(x_test)

    return x_usv_train, x_sv_train, y_sv_train, x_test, y_test


def with_ccfraud(x_ben, x_fraud, usv_train, sv_train, sv_train_fraud, test_fraud, test_benign, cross_validation_k):
    sv_train_ben = sv_train - sv_train_fraud

    x_usv_train, x_sv_train, y_sv_train, x_test, y_test = \
        data_sampling(x_ben, x_fraud, usv_train, sv_train_ben, sv_train_fraud, test_fraud, test_benign,
                      cross_validation_k)

    sc = MinMaxScaler()
    if len(x_sv_train) > len(x_usv_train):
        x_sv_train = sc.fit_transform(x_sv_train)
        x_usv_train = sc.transform(x_usv_train)
    else:
        x_usv_train = sc.fit_transform(x_usv_train)
        x_sv_train = sc.transform(x_sv_train)

    x_test = sc.transform(x_test)

    return x_usv_train, x_sv_train, y_sv_train, x_test, y_test


def with_ieee(x_ben, x_fraud, usv_train, sv_train, sv_train_fraud, test_fraud, test_benign, cross_validation_k):
    sv_train_ben = sv_train - sv_train_fraud

    x_usv_train, x_sv_train, y_sv_train, x_test, y_test = \
        data_sampling(x_ben, x_fraud, usv_train, sv_train_ben, sv_train_fraud, test_fraud, test_benign,
                      cross_validation_k)

    # Cleaning infinite values to NaN
    x_usv_train = clean_inf_nan(x_usv_train)
    x_sv_train = clean_inf_nan(x_sv_train)
    x_test = clean_inf_nan(x_test)

    pca = PCA(n_components=x_usv_train.shape[1])
    if len(x_sv_train) > len(x_usv_train):
        x_sv_train = pca.fit_transform(X=x_sv_train)
        x_usv_train = pca.transform(X=x_usv_train)
    else:
        x_usv_train = pca.fit_transform(x_usv_train)
        x_sv_train = pca.transform(x_sv_train)
    x_test = pca.transform(X=x_test)

    sc = StandardScaler()
    if len(x_sv_train) > len(x_usv_train):
        x_sv_train = sc.fit_transform(x_sv_train)
        x_usv_train = sc.transform(x_usv_train)
    else:
        x_usv_train = sc.fit_transform(x_usv_train)
        x_sv_train = sc.transform(x_sv_train)
    x_test = sc.transform(x_test)

    return x_usv_train, x_sv_train, y_sv_train, x_test, y_test


def data_sampling(x_ben, x_fraud, usv_train, sv_train_ben, sv_train_fraud, test_fraud, test_benign, cross_validation_k):
    k = cross_validation_k
    # Take random sample of sufficient space (including some offset)
    x_ben = x_ben.sample(n=k * (usv_train + sv_train_ben + sv_train_fraud + test_benign + test_fraud)).values
    x_fraud = x_fraud.sample(frac=1).values

    x_usv_train = x_ben[0:k * usv_train]
    x_sv_train_ben = x_ben[0:k * sv_train_ben]
    x_sv_train_fraud = x_fraud[0: k * sv_train_fraud]
    x_y_sv_train_ben = np.append(x_sv_train_ben, np.zeros((k * sv_train_ben, 1)), axis=1)
    x_y_sv_train_fraud = np.append(x_sv_train_fraud, np.ones((k * sv_train_fraud, 1)), axis=1)
    x_y_sv_train = np.concatenate((x_y_sv_train_ben, x_y_sv_train_fraud))
    x_y_sv_train = sample_shuffle(x_y_sv_train)
    x_sv_train = x_y_sv_train[:, :-1]
    y_sv_train = x_y_sv_train[:, -1]

    x_test = x_ben[-test_benign:].tolist() + x_fraud[-test_fraud:].tolist()
    x_test = np.array(x_test)

    y_test = np.zeros((test_benign + test_fraud))
    y_test[test_benign:] = 1

    return x_usv_train, x_sv_train, y_sv_train, x_test, y_test


def execute_smote(x_sv_train, y_sv_train):
    sm = SMOTE()
    x_res, y_res = sm.fit_resample(X=x_sv_train, y=y_sv_train)

    return x_res, y_res
