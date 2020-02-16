import argparse
from argparse import RawTextHelpFormatter
import numpy as np
from tabulate import tabulate

from baselines.calculate_baselines import build_classic_baselines
from baselines.calculate_oc_baselines import build_oc_baselines
from utils.list_operations import sample_shuffle
from utils.load_data import get_data_paysim, get_data_ccfraud, get_data_ieee
from utils.sample_data import sample_data_for_occ, sample_data_for_normal_classification

datasets = ["paysim", "ccfraud", "ieee"]

parser = argparse.ArgumentParser(description='Tool for testing various machine learning methods on different datasets',
                                 formatter_class=RawTextHelpFormatter)
parser.add_argument("--dataset", required=True, choices=datasets, help="Dataset")
parser.add_argument("--method", choices=["sk-svm", "sk-knn", "sk-mlp", "sk-nb", "sk-rf", "sk-lr", "xgb"],
                    help="Machine learning method used for classification")
parser.add_argument("--mode", choices=["baseline", "solo"], help='''Execution mode: 
`baseline` for comparison to other baseline methods
`solo` for executing the chosen method only''')

args = parser.parse_args()
dataset_string = args.dataset

if dataset_string == "paysim":
    x_ben, x_fraud = get_data_paysim("paysim.csv")
    benign_fraud_ratio = len(x_ben) / (len(x_ben) + len(x_fraud))
    x_ben = sample_shuffle(x_ben)[0:2000]
elif dataset_string == "ccfraud":
    x_ben, x_fraud = get_data_ccfraud("ccfraud.csv")
    benign_fraud_ratio = len(x_ben) / (len(x_ben) + len(x_fraud))
    x_ben = sample_shuffle(x_ben)[0:2000]
elif dataset_string == "ieee":
    x_ben, x_fraud = get_data_ieee("ieee.csv")
    benign_fraud_ratio = len(x_ben) / (len(x_ben) + len(x_fraud))
    x_ben = sample_shuffle(x_ben)[0:2000]

occ_train_size = 700
train_test_ratio = 0.8

baseline_train_size = 1000
baseline_negative_samples = 14

iteration_count = 10

prec_coll = list()
reca_coll = list()
f1_coll = list()
acc_coll = list()

occ_methods = ['OC-SVM', 'Elliptic Envelope', 'Isolation Forest']
baseline_methods = ['SVM SVC', 'kNN', 'Decision Tree', 'Random Forest', 'SVM Linear SVC', 'Gaussian NB',
                    'Logistic Regression', 'XG Boost']

for i in range(iteration_count):
    # One-Class Classification
    if dataset_string == "paysim":
        print(dataset_string)
    elif dataset_string == "ccfraud":
        x_train, x_test, y_train, y_test = sample_data_for_occ(x_ben, x_fraud, dataset_string)
    elif dataset_string == "ieee":
        print(dataset_string)

    prec_oc_list, reca_oc_list, f1_oc_list, acc_oc_list = build_oc_baselines(x_train, x_test, y_test, occ_train_size)

    # Normal classification
    if dataset_string == "paysim":
        print(dataset_string)
    elif dataset_string == "ccfraud":
        x_train, x_test, y_train, y_test = \
            sample_data_for_normal_classification(x_ben, x_fraud, baseline_train_size, baseline_negative_samples,
                                                  dataset_string)
    elif dataset_string == "ieee":
        print(dataset_string)

    prec_list, reca_list, f1_list, acc_list = build_classic_baselines(x_train, y_train, x_test, y_test)

    # Add metrics for all methods to collections
    prec_coll.append(prec_oc_list + prec_list)
    reca_coll.append(reca_oc_list + reca_list)
    f1_coll.append(f1_oc_list + f1_list)
    acc_coll.append(acc_oc_list + acc_list)


prec_coll, reca_coll, f1_coll, acc_coll = \
    np.array(prec_coll), np.array(reca_coll), np.array(f1_coll), np.array(acc_coll)

print(f'Average metrics over {iteration_count} iterations')

results = list()
methods = occ_methods + baseline_methods

for index, method in enumerate(methods):
    prec = f'{np.mean(prec_coll[:, index]).round(3)} \u00B1 {np.std(prec_coll[:, index]).round(3)}'
    reca = f'{np.mean(reca_coll[:, index]).round(3)} \u00B1 {np.std(reca_coll[:, index]).round(3)}'
    f1 = f'{np.mean(f1_coll[:, index]).round(3)} \u00B1 {np.std(f1_coll[:, index]).round(3)}'
    acc = f'{np.mean(acc_coll[:, index]).round(3)} \u00B1 {np.std(acc_coll[:, index]).round(3)}'
    results.append([method, prec, reca, f1, acc])

print(tabulate(results, headers=['Method', 'Precision', 'Recall', 'F1 score', 'Accuracy']))

