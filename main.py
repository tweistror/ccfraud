import argparse
from argparse import RawTextHelpFormatter
import numpy as np
from tabulate import tabulate
from datetime import datetime

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
parser.add_argument("--v", choices=['0', '1', '2'], default=0, help="Specify verbosity")

args = parser.parse_args()
dataset_string = args.dataset
verbosity = int(args.v)

# Specify positive samples to load
positive_samples = 40000

if dataset_string == "paysim":
    x_ben, x_fraud = get_data_paysim("paysim.csv")
    unsupervised_train_size = 5000
    supervised_train_size = 2000
    supervised_train_negative_samples = 50
    test_negative_samples = 1000
    x_ben = sample_shuffle(x_ben)
elif dataset_string == "ccfraud":
    x_ben, x_fraud = get_data_ccfraud("ccfraud.csv")
    unsupervised_train_size = 700
    supervised_train_size = 1000
    supervised_train_negative_samples = 10
    test_negative_samples = 490
    x_ben = sample_shuffle(x_ben)
elif dataset_string == "ieee":
    x_ben, x_fraud = get_data_ieee("ieee.csv")
    unsupervised_train_size = 700
    x_ben = sample_shuffle(x_ben)


iteration_count = 10

prec_coll = list()
reca_coll = list()
f1_coll = list()
acc_coll = list()

occ_methods = ['OC-SVM', 'Elliptic Envelope', 'Isolation Forest', 'kNN Local Outlier Factor']
baseline_methods = ['SVM SVC', 'kNN', 'Decision Tree', 'Random Forest', 'SVM Linear SVC', 'Gaussian NB',
                    'Logistic Regression', 'XG Boost', 'SGD', 'Gaussian Process', 'Decision Tree', 'Adaboost']

start_time_complete = datetime.now()
if verbosity > 0:
    print(f'Start {iteration_count} iterations')

for i in range(iteration_count):
    start_time = datetime.now()
    if verbosity > 1:
        print(f'Starting iteration #{i+1}')


    # Sampe for One-Class Classification
    x_train, x_test, y_train, y_test = sample_data_for_occ(x_ben, x_fraud)

    prec_oc_list, reca_oc_list, f1_oc_list, acc_oc_list = build_oc_baselines(x_train, x_test, y_test,
                                                                             unsupervised_train_size, test_negative_samples)

    # Some verbosity output
    if verbosity > 1:
        print(f'Iteration #{i+1} unsupervised finished, supervised coming up')

    # Normal classification

    x_train, x_test, y_train, y_test = \
        sample_data_for_normal_classification(x_ben, x_fraud, supervised_train_size, supervised_train_negative_samples)

    prec_list, reca_list, f1_list, acc_list = build_classic_baselines(x_train, y_train, x_test, y_test, test_negative_samples)

    # Add metrics for all methods to collections
    prec_coll.append(prec_oc_list + prec_list)
    reca_coll.append(reca_oc_list + reca_list)
    f1_coll.append(f1_oc_list + f1_list)
    acc_coll.append(acc_oc_list + acc_list)

    if verbosity > 0:
        time_required = str(datetime.now() - start_time)
        print(f'Iteration #{i+1} finished in {time_required}')


if verbosity > 1:
    time_required = str(datetime.now() - start_time_complete)
    print(f'All {iteration_count} iterations finished in {time_required}')

prec_coll, reca_coll, f1_coll, acc_coll = \
    np.array(prec_coll), np.array(reca_coll), np.array(f1_coll), np.array(acc_coll)

print(f'Average metrics over {iteration_count} iterations')

results = list()
methods = occ_methods + baseline_methods

for index, method in enumerate(methods):
    if index == 0:
        results.append(['Unsupervised Learning Methods'])

    if index == len(occ_methods):
        results.append(['Supervised Learning Methods'])

    prec = f'{np.mean(prec_coll[:, index]).round(3)} \u00B1 {np.std(prec_coll[:, index]).round(3)}'
    reca = f'{np.mean(reca_coll[:, index]).round(3)} \u00B1 {np.std(reca_coll[:, index]).round(3)}'
    f1 = f'{np.mean(f1_coll[:, index]).round(3)} \u00B1 {np.std(f1_coll[:, index]).round(3)}'
    acc = f'{np.mean(acc_coll[:, index]).round(3)} \u00B1 {np.std(acc_coll[:, index]).round(3)}'
    results.append([method, prec, reca, f1, acc])

print(tabulate(results, headers=['Method', 'Precision', 'Recall', 'F1 score', 'Accuracy']))

