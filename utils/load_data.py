from datetime import datetime

import pandas as pd
import numpy as np
from mlxtend.data import loadlocal_mnist

from utils.data_loading.saperp_synthetic import get_data_saperp
from utils.list_operations import sample_shuffle
from utils.preprocessing.ccfraud import Preprocess_ccfraud
from utils.preprocessing.cifar10 import Preprocess_cifar10
from utils.preprocessing.ieee import Preprocess_ieee
from utils.preprocessing.mnist import Preprocess_mnist
from utils.preprocessing.nslkdd import Preprocess_nslkdd
from utils.preprocessing.paysim import Preprocess_paysim
from utils.preprocessing.paysim_custom import Preprocess_paysim_custom
from utils.preprocessing.utils import drop_columns, one_hot_encode_column, unpickle


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
            fraud_only = self.parameter_class.get_saperp_mode()['fraud_only']
            x_ben, x_fraud = get_data_saperp(self.dataset_string, self.path, fraud_only)
        elif self.dataset_string == "mnist":
            x_ben, x_fraud, preprocessing_class = self.get_data_mnist()
        elif self.dataset_string == "cifar10":
            x_ben, x_fraud, preprocessing_class = self.get_data_cifar10()

        if self.dataset_string == "cifar10" or self.dataset_string == "mnist":
            x_ben = sample_shuffle(x_ben, self.seed)
            x_fraud = sample_shuffle(x_fraud, self.seed)
        else:
            preprocessing_class.set_columns(x_ben)
            x_ben = x_ben.sample(frac=1, random_state=self.seed).values
            x_fraud = x_fraud.sample(frac=1, random_state=self.seed).values

        return x_ben, x_fraud, preprocessing_class

    def get_data_paysim(self):
        pp_paysim = Preprocess_paysim()

        data = self.read_csv(self.path['one'])

        # Add feature for `nameOrig` to `nameDest` relation with one-hot encoding
        #  => Feature is not important
        # data['nameOrig'] = data['nameOrig'].apply(lambda x: x[:1])
        # data['nameDest'] = data['nameDest'].apply(lambda x: x[:1])
        # data['from_to'] = data['nameOrig'] + data['nameDest']
        # data = pd.concat([data, pd.get_dummies(data['from_to'], prefix='from_to')], axis=1)
        # data.drop(columns=['from_to'], inplace=True)

        x_ben, x_fraud = pp_paysim.initial_processing(data)

        return x_ben, x_fraud, pp_paysim

    def get_data_ccfraud(self):
        pp_ccfraud = Preprocess_ccfraud()

        data = self.read_csv(self.path['one'])

        x_ben, x_fraud = pp_ccfraud.initial_processing(data)

        return x_ben, x_fraud, pp_ccfraud

    def get_data_paysim_custom(self):
        pp_paysim_custom = Preprocess_paysim_custom()

        data = self.read_csv(self.path['one'])

        x_ben, x_fraud = pp_paysim_custom.initial_processing(data)

        return x_ben, x_fraud, pp_paysim_custom

    def get_data_ieee(self):
        pp_ieee = Preprocess_ieee()

        # if skip is True:
        #     start_time = datetime.now()
        #     x_ben = pd.read_csv('./debug/ieee/x_ben.csv')
        #     x_fraud = pd.read_csv('./debug/ieee/x_fraud.csv')
        #     if verbosity > 0:
        #         print(f'IEEE: Preprocessed dataset loaded in {str(datetime.now() - start_time)}')
        #
        #     return x_ben, x_fraud

        transaction_data = self.read_csv(self.path['one'])
        identity_data = self.read_csv(self.path['two'])

        data = pd.merge(transaction_data, identity_data, on='TransactionID', how='left')
        del transaction_data, identity_data

        x_ben, x_fraud = pp_ieee.initial_processing(data)

        # x_ben.to_csv(r'x_ben.csv')
        # x_fraud.to_csv(r'x_fraud.csv')

        return x_ben, x_fraud, pp_ieee

    def get_data_nslkdd(self):
        pp_nslkdd = Preprocess_nslkdd()

        columns = ["duration", "protocol_type", "service", "flag", "src_bytes",
                   "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                   "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                   "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                   "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                   "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                   "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                   "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                   "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                   "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"]

        train_data = self.read_csv(self.path['one'], columns=columns)
        test_data = self.read_csv(self.path['two'], columns=columns)

        data = train_data.append(test_data)
        del train_data, test_data

        x_ben, x_fraud = pp_nslkdd.initial_processing(data)

        return x_ben, x_fraud, pp_nslkdd

    def get_data_cifar10(self):
        anomaly_number = self.parameter_class.get_mnist_mode()['anomaly_number']
        pp_cifar10 = Preprocess_cifar10(anomaly_number)

        labels = None
        images = None

        for i in range(1, 7):
            batch_path = self.path[f'batch{i}']
            batch = unpickle(batch_path)
            if i == 1:
                labels = batch[b'labels']
                images = batch[b'data']
            else:
                labels = np.concatenate((labels, batch[b'labels']))
                images = np.concatenate((images, batch[b'data']))

        x_ben, x_fraud = pp_cifar10.initial_processing(images, labels)

        return x_ben, x_fraud, pp_cifar10

    def get_data_mnist(self):
        anomaly_number = self.parameter_class.get_mnist_mode()['anomaly_number']
        pp_mnist = Preprocess_mnist(anomaly_number)

        train_images, train_labels = loadlocal_mnist(
            images_path=self.path['train_images'],
            labels_path=self.path['train_labels'])
        test_images, test_labels = loadlocal_mnist(
            images_path=self.path['test_images'],
            labels_path=self.path['test_labels'])

        x_ben, x_fraud = pp_mnist.initial_processing(train_images, train_labels, test_images, test_labels)

        return x_ben, x_fraud, pp_mnist


    def read_csv(self, path, columns=None):
        start_time = datetime.now()
        if self.verbosity > 0:
            print(f'{path}: Start loading dataset')

        data = pd.read_csv(f'{path}', names=columns)

        if self.verbosity > 0:
            time_required = str(datetime.now() - start_time)
            print(f'{path}: Dataset loaded in {time_required}')

        return data
