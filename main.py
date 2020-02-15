import argparse
from argparse import RawTextHelpFormatter
import numpy as np
from tabulate import tabulate

from baselines.sklearn_baselines import svm_oneclass, svm_svc
from utils.list_operations import sample_shuffle
from utils.load_data import get_data_paysim, get_data_ccfraud, get_data_ieee
from utils.run_models import run_one_svm, run_svm
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
baseline_negative_samples = 10

iteration_count = 10

prec_coll_occ = list()
reca_coll_occ = list()
f1_coll_occ = list()
acc_coll_occ = list()

occ_methods = ['OC-SVM']

# One-Class Classification
for i in range(iteration_count):
    if dataset_string == "paysim":
        print(dataset_string)
    elif dataset_string == "ccfraud":
        x_train, x_test, y_train, y_test = sample_data_for_occ(x_ben, x_fraud, dataset_string)
    elif dataset_string == "ieee":
        print(dataset_string)

    # OC-SVM
    clf = svm_oneclass(x_train[0:occ_train_size])
    prec_svm, reca_svm, f1_svm, acc_svm = run_one_svm(x_test, y_test, clf, 'fraud-prediction')

    # Add metrics for all one-class methods to collections
    prec_coll_occ.append([prec_svm])
    reca_coll_occ.append([reca_svm])
    f1_coll_occ.append([f1_svm])
    acc_coll_occ.append([acc_svm])

prec_coll_bc = list()
reca_coll_bc = list()
f1_coll_bc = list()
acc_coll_bc = list()

baseline_methods = ['SVM']

# Baseline standard classification
for i in range(iteration_count):
    if dataset_string == "paysim":
        print(dataset_string)
    elif dataset_string == "ccfraud":
        x_train, x_test, y_train, y_test = \
            sample_data_for_normal_classification(x_ben, x_fraud, baseline_train_size, baseline_negative_samples,
                                                  dataset_string)
    elif dataset_string == "ieee":
        print(dataset_string)

    # SVM
    clf = svm_svc(x_train, y_train)
    prec_svm, reca_svm, f1_svm, acc_svm = run_svm(x_test, y_test, clf, 'fraud-prediction')

    # Add metrics for all one-class methods to collections
    prec_coll_bc.append([prec_svm])
    reca_coll_bc.append([reca_svm])
    f1_coll_bc.append([f1_svm])
    acc_coll_bc.append([acc_svm])


# OCC metrics
prec_coll_occ, reca_coll_occ, f1_coll_occ, acc_coll_occ = \
    np.array(prec_coll_occ), np.array(reca_coll_occ), np.array(f1_coll_occ), np.array(acc_coll_occ)

# BC metrics
prec_coll_bc, reca_coll_bc, f1_coll_bc, acc_coll_bc = \
    np.array(prec_coll_bc), np.array(reca_coll_bc), np.array(f1_coll_bc), np.array(acc_coll_bc)

print(f'Average metrics over {iteration_count} iterations')
for index, method in enumerate(occ_methods):
    prec = f'{np.mean(prec_coll_occ[:, index]).round(3)} \u00B1 {np.std(prec_coll_occ[:, index]).round(3)}'
    reca = f'{np.mean(reca_coll_occ[:, index]).round(3)} \u00B1 {np.std(reca_coll_occ[:, index]).round(3)}'
    f1 = f'{np.mean(f1_coll_occ[:, index]).round(3)} \u00B1 {np.std(f1_coll_occ[:, index]).round(3)}'
    acc = f'{np.mean(acc_coll_occ[:, index]).round(3)} \u00B1 {np.std(acc_coll_occ[:, index]).round(3)}'

    print(tabulate([[method, prec, reca, f1, acc]], headers=['Method', 'Precision', 'Recall', 'F1 score', 'Accuracy']))

print(f'Average metrics over {iteration_count} iterations')
for index, method in enumerate(baseline_methods):
    prec = f'{np.mean(prec_coll_bc[:, index]).round(3)} \u00B1 {np.std(prec_coll_bc[:, index]).round(3)}'
    reca = f'{np.mean(reca_coll_bc[:, index]).round(3)} \u00B1 {np.std(reca_coll_bc[:, index]).round(3)}'
    f1 = f'{np.mean(f1_coll_bc[:, index]).round(3)} \u00B1 {np.std(f1_coll_bc[:, index]).round(3)}'
    acc = f'{np.mean(acc_coll_bc[:, index]).round(3)} \u00B1 {np.std(acc_coll_bc[:, index]).round(3)}'

    print(tabulate([[method, prec, reca, f1, acc]], headers=['Method', 'Precision', 'Recall', 'F1 score', 'Accuracy']))
