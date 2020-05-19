import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils.list_operations import sample_shuffle, clean_inf_nan
from utils.preprocessing.utils import is_image_dataset


class SplitPreprocessData(object):
    def __init__(self, dataset_string, preprocess_class, seed, cross_validation_k=0, verbosity=0):
        self.dataset_string = dataset_string
        self.preprocess_class = preprocess_class
        self.seed = seed
        self.cross_validation_k = cross_validation_k
        self.verbosity = verbosity

        self.scaler = None
        self.pca = None

        self.x_usv_train = None
        self.x_sv_train = None
        self.y_sv_train = None
        self.x_test = None
        self.y_test = None

    def execute_split_preprocess(self, x_ben, x_fraud, usv_train, sv_train, sv_train_fraud, test_fraud, test_benign):
        parameters = [x_ben, x_fraud, usv_train, sv_train, sv_train_fraud, test_fraud, test_benign]
        parameter_strings = ['x_ben', 'x_fraud', 'usv_train', 'sv_train', 'sv_train_fraud', 'test_fraud', 'test_benign']

        parameter_dict = {parameter_strings[i]: parameters[i] for i in range(0, len(parameters))}
        parameter_dict['sv_train_ben'] = sv_train - sv_train_fraud

        self.x_usv_train, self.x_sv_train, self.y_sv_train, self.x_test, self.y_test = \
            self.split_data(parameter_dict)

        if self.dataset_string == "paysim" or self.dataset_string == "paysim-custom":
            x_usv_train, x_sv_train, x_test = self.with_paysim()
        elif self.dataset_string == "ccfraud":
            x_usv_train, x_sv_train, x_test = self.with_ccfraud()
        elif self.dataset_string == "ieee":
            x_usv_train, x_sv_train, x_test = self.with_ieee()
        elif self.dataset_string == "nslkdd":
            x_usv_train, x_sv_train, x_test = self.with_nslkdd()
        elif self.dataset_string == "saperp-ek" or self.dataset_string == "saperp-vk":
            x_usv_train, x_sv_train, x_test = self.with_saperp()
        elif self.dataset_string == "mnist":
            x_usv_train, x_sv_train, x_test = self.with_mnist()
        elif self.dataset_string == "cifar10":
            x_usv_train, x_sv_train, x_test = self.with_cifar10()

        return x_usv_train, x_sv_train, self.y_sv_train, x_test, self.y_test

    def with_paysim(self):
        pp_paysim = self.preprocess_class

        x_sv_train, x_usv_train, x_test = pp_paysim.preprocess(self.x_sv_train, self.x_usv_train, self.x_test)

        # print(pp_paysim.inverse_preprocessing(x_sv_train).head(5))

        return x_usv_train, x_sv_train, x_test

    def with_ccfraud(self):
        pp_ccfraud = self.preprocess_class

        x_sv_train, x_usv_train, x_test = pp_ccfraud.preprocess(self.x_sv_train, self.x_usv_train, self.x_test)

        return x_usv_train, x_sv_train, x_test

    def with_ieee(self):
        pp_ieee = self.preprocess_class

        x_sv_train, x_usv_train, x_test = pp_ieee.preprocess(self.x_sv_train, self.x_usv_train, self.x_test)

        # print(pp_ieee.inverse_preprocessing(x_sv_train).head(10))

        return x_usv_train, x_sv_train, x_test

    def with_nslkdd(self):
        pp_nslkdd = self.preprocess_class

        x_sv_train, x_usv_train, x_test = pp_nslkdd.preprocess(self.x_sv_train, self.x_usv_train, self.x_test)

        # print(pp_nslkdd.inverse_preprocessing(x_sv_train).head(10))

        return x_usv_train, x_sv_train, x_test

    def with_saperp(self, parameters):
        x_usv_train, x_sv_train, y_sv_train, x_test, y_test = \
            self.split_data(parameters)

        return x_usv_train, x_sv_train, y_sv_train, x_test, y_test

    def with_mnist(self):
        pp_mnist = self.preprocess_class

        x_sv_train, x_usv_train, x_test = pp_mnist.preprocess(self.x_sv_train, self.x_usv_train, self.x_test)

        return x_usv_train, x_sv_train, x_test

    def with_cifar10(self):
        pp_cifar10 = self.preprocess_class

        x_sv_train, x_usv_train, x_test = pp_cifar10.preprocess(self.x_sv_train, self.x_usv_train, self.x_test)

        return x_usv_train, x_sv_train, x_test

    def split_data(self, parameters):
        k = self.cross_validation_k
        usv_train = parameters['usv_train']
        sv_train_ben = parameters['sv_train_ben']
        sv_train_fraud = parameters['sv_train_fraud']
        test_benign = parameters['test_benign']
        test_fraud = parameters['test_fraud']
        x_ben = parameters['x_ben']
        x_fraud = parameters['x_fraud']

        x_usv_train = x_ben[0:k * usv_train]
        x_sv_train_ben = x_ben[0:k * sv_train_ben]
        x_sv_train_fraud = x_fraud[0: k * sv_train_fraud]
        x_y_sv_train_ben = np.append(x_sv_train_ben, np.zeros((k * sv_train_ben, 1)), axis=1)
        x_y_sv_train_fraud = np.append(x_sv_train_fraud, np.ones((k * sv_train_fraud, 1)), axis=1)
        x_y_sv_train = np.concatenate((x_y_sv_train_ben, x_y_sv_train_fraud))
        x_y_sv_train = sample_shuffle(x_y_sv_train, self.seed)
        x_sv_train = x_y_sv_train[:, :-1]
        y_sv_train = x_y_sv_train[:, -1]

        x_test = x_ben[-test_benign:].tolist() + x_fraud[-test_fraud:].tolist()
        x_test = np.array(x_test)

        y_test = np.zeros((test_benign + test_fraud))
        y_test[test_benign:] = 1

        return x_usv_train, x_sv_train, y_sv_train, x_test, y_test


