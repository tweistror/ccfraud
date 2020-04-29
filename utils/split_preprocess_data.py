import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.list_operations import sample_shuffle, clean_inf_nan


class SplitPreprocessData(object):
    def __init__(self, dataset_string, cross_validation_k=0, verbosity=0):
        self.dataset_string = dataset_string
        self.cross_validation_k = cross_validation_k
        self.verbosity = verbosity

    def execute_split_preprocess(self, x_ben, x_fraud, usv_train, sv_train, sv_train_fraud, test_fraud, test_benign):
        parameters = [x_ben, x_fraud, usv_train, sv_train, sv_train_fraud, test_fraud, test_benign]
        parameter_strings = ['x_ben', 'x_fraud', 'usv_train', 'sv_train', 'sv_train_fraud', 'test_fraud', 'test_benign']

        parameter_dict = {parameter_strings[i]: parameters[i] for i in range(0, len(parameters))}
        parameter_dict['sv_train_ben'] = sv_train - sv_train_fraud

        if self.dataset_string == "paysim" or self.dataset_string == "paysim-custom":
            x_usv_train, x_sv_train, y_sv_train, x_test, y_test = self.with_paysim(parameter_dict)
        elif self.dataset_string == "ccfraud":
            x_usv_train, x_sv_train, y_sv_train, x_test, y_test = self.with_ccfraud(parameter_dict)
        elif self.dataset_string == "ieee":
            x_usv_train, x_sv_train, y_sv_train, x_test, y_test = self.with_ieee(parameter_dict)
        elif self.dataset_string == "nslkdd":
            x_usv_train, x_sv_train, y_sv_train, x_test, y_test = self.with_nslkdd(parameter_dict)
        elif self.dataset_string == "saperp-ek" or self.dataset_string == "saperp-vk":
            x_usv_train, x_sv_train, y_sv_train, x_test, y_test = self.with_saperp(parameter_dict)

        return x_usv_train, x_sv_train, y_sv_train, x_test, y_test

    def with_paysim(self, parameters):
        x_usv_train, x_sv_train, y_sv_train, x_test, y_test = \
            self.data_sampling(parameters)

        pca = PCA(n_components=x_usv_train.shape[1])
        if len(x_sv_train) > len(x_usv_train):
            x_sv_train = pca.fit_transform(X=x_sv_train)
            x_usv_train = pca.transform(X=x_usv_train)
        else:
            x_usv_train = pca.fit_transform(x_usv_train)
            x_sv_train = pca.transform(x_sv_train)
        x_test = pca.transform(X=x_test)

        sc = MinMaxScaler()
        if len(x_sv_train) > len(x_usv_train):
            x_sv_train = sc.fit_transform(x_sv_train)
            x_usv_train = sc.transform(x_usv_train)
        else:
            x_usv_train = sc.fit_transform(x_usv_train)
            x_sv_train = sc.transform(x_sv_train)
        x_test = sc.transform(x_test)

        return x_usv_train, x_sv_train, y_sv_train, x_test, y_test

    def with_ccfraud(self, parameters):
        x_usv_train, x_sv_train, y_sv_train, x_test, y_test = \
            self.data_sampling(parameters)

        sc = MinMaxScaler()
        if len(x_sv_train) > len(x_usv_train):
            x_sv_train = sc.fit_transform(x_sv_train)
            x_usv_train = sc.transform(x_usv_train)
        else:
            x_usv_train = sc.fit_transform(x_usv_train)
            x_sv_train = sc.transform(x_sv_train)

        x_test = sc.transform(x_test)

        return x_usv_train, x_sv_train, y_sv_train, x_test, y_test

    def with_ieee(self, parameters):
        x_usv_train, x_sv_train, y_sv_train, x_test, y_test = \
            self.data_sampling(parameters)

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

    def with_nslkdd(self, parameters):
        x_usv_train, x_sv_train, y_sv_train, x_test, y_test = \
            self.data_sampling(parameters)

        sc = MinMaxScaler()
        if len(x_sv_train) > len(x_usv_train):
            x_sv_train = sc.fit_transform(x_sv_train)
            x_usv_train = sc.transform(x_usv_train)
        else:
            x_usv_train = sc.fit_transform(x_usv_train)
            x_sv_train = sc.transform(x_sv_train)

        x_test = sc.transform(x_test)

        return x_usv_train, x_sv_train, y_sv_train, x_test, y_test

    def with_saperp(self, parameters):
        x_usv_train, x_sv_train, y_sv_train, x_test, y_test = \
            self.data_sampling(parameters)

        return x_usv_train, x_sv_train, y_sv_train, x_test, y_test

    def data_sampling(self, parameters):
        k = self.cross_validation_k
        usv_train = parameters['usv_train']
        sv_train_ben = parameters['sv_train_ben']
        sv_train_fraud = parameters['sv_train_fraud']
        test_benign = parameters['test_benign']
        test_fraud = parameters['test_fraud']
        x_ben = parameters['x_ben']
        x_fraud = parameters['x_fraud']

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
