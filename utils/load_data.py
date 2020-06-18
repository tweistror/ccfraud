from utils.preprocessing.ccfraud import Preprocess_ccfraud
from utils.preprocessing.cifar10 import Preprocess_cifar10
from utils.preprocessing.ieee import Preprocess_ieee
from utils.preprocessing.mnist import Preprocess_mnist
from utils.preprocessing.nslkdd import Preprocess_nslkdd
from utils.preprocessing.paysim import Preprocess_paysim
from utils.preprocessing.paysim_custom import Preprocess_paysim_custom
from utils.preprocessing.saperp import Preprocess_saperp


class LoadData(object):
    def __init__(self, dataset_string, path, seed, parameter_class, verbosity=0):
        self.dataset_string = dataset_string
        self.verbosity = verbosity
        self.path = path
        self.seed = seed
        self.parameter_class = parameter_class

    def get_data(self):
        x_ben = None
        x_fraud = None
        preprocessing_class = None

        if self.dataset_string == "paysim":
            x_ben, x_fraud, preprocessing_class = self.get_data_paysim()
        if self.dataset_string == "paysim-custom":
            x_ben, x_fraud, preprocessing_class = self.get_data_paysim_custom()
        elif self.dataset_string == "ccfraud":
            x_ben, x_fraud, preprocessing_class = self.get_data_ccfraud()
        elif self.dataset_string == "ieee":
            x_ben, x_fraud, preprocessing_class = self.get_data_ieee()
        elif self.dataset_string == "nslkdd":
            x_ben, x_fraud, preprocessing_class = self.get_data_nslkdd()
        elif self.dataset_string == "saperp-ek" or self.dataset_string == "saperp-vk":
            x_ben, x_fraud, preprocessing_class = self.get_data_saperp()
        elif self.dataset_string == "mnist":
            x_ben, x_fraud, preprocessing_class = self.get_data_mnist()
        elif self.dataset_string == "cifar10":
            x_ben, x_fraud, preprocessing_class = self.get_data_cifar10()

        if self.dataset_string != "cifar10" and self.dataset_string != "mnist":
            preprocessing_class.set_columns(x_ben)
            x_ben = x_ben.values
            x_fraud = x_fraud.values

        return x_ben, x_fraud, preprocessing_class

    def get_data_paysim(self):
        pp_paysim = Preprocess_paysim(self.path)

        x_ben, x_fraud = pp_paysim.initial_processing()

        return x_ben, x_fraud, pp_paysim

    def get_data_ccfraud(self):
        pp_ccfraud = Preprocess_ccfraud(self.path)

        x_ben, x_fraud = pp_ccfraud.initial_processing()

        return x_ben, x_fraud, pp_ccfraud

    def get_data_paysim_custom(self):
        pp_paysim_custom = Preprocess_paysim_custom(self.path)

        x_ben, x_fraud = pp_paysim_custom.initial_processing()

        return x_ben, x_fraud, pp_paysim_custom

    def get_data_ieee(self):
        pp_ieee = Preprocess_ieee(self.path)

        x_ben, x_fraud = pp_ieee.initial_processing()

        return x_ben, x_fraud, pp_ieee

    def get_data_saperp(self):
        fraud_only = self.parameter_class.get_saperp_mode()['fraud_only']
        pp_saperp = Preprocess_saperp(self.dataset_string, self.path, fraud_only)

        x_ben, x_fraud = pp_saperp.initial_processing()

        return x_ben, x_fraud, pp_saperp

    def get_data_nslkdd(self):
        pp_nslkdd = Preprocess_nslkdd(self.path)

        x_ben, x_fraud = pp_nslkdd.initial_processing()

        return x_ben, x_fraud, pp_nslkdd

    def get_data_cifar10(self):
        anomaly_number = self.parameter_class.get_cifar10_mode()['anomaly_number']
        train_mode = self.parameter_class.get_cifar10_mode()['train_mode']

        pp_cifar10 = Preprocess_cifar10(self.path, anomaly_number, train_mode)

        x_ben, x_fraud = pp_cifar10.initial_processing()

        return x_ben, x_fraud, pp_cifar10

    def get_data_mnist(self):
        anomaly_number = self.parameter_class.get_mnist_mode()['anomaly_number']
        train_mode = self.parameter_class.get_mnist_mode()['train_mode']

        pp_mnist = Preprocess_mnist(self.path, anomaly_number, train_mode)

        x_ben, x_fraud = pp_mnist.initial_processing()

        return x_ben, x_fraud, pp_mnist


