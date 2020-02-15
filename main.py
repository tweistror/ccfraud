import argparse
from argparse import RawTextHelpFormatter
import numpy as np
from tabulate import tabulate

from baselines.sklearn_baselines import svm_oneclass
from utils.list_operations import sample_shuffle
from utils.load_data import get_data_paysim, get_data_ccfraud, get_data_ieee
from utils.run_models import run_one_svm
from utils.sample_data import sample_data_for_occ

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

sample_size = 700

if dataset_string == "paysim":
    x_ben, x_fraud = get_data_paysim("paysim.csv")
    x_ben = sample_shuffle(x_ben)[0:2000]
elif dataset_string == "ccfraud":
    x_ben, x_fraud = get_data_ccfraud("ccfraud.csv")
    x_ben = sample_shuffle(x_ben)[0:2000]
elif dataset_string == "ieee":
    x_ben, x_fraud = get_data_ieee("ieee.csv")
    x_ben = sample_shuffle(x_ben)[0:2000]

train_size = 700
train_test_ratio = 0.75

iteration_count = 10


prec_coll = list()
reca_coll = list()
f1_coll = list()
acc_coll = list()

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
    clf = svm_oneclass(x_train[0:train_size])
    prec_svm, reca_svm, f1_svm, acc_svm = run_one_svm(x_test, y_test, clf, 'fraud-prediction')

    # Add metrics for all one-class methods to collections
    prec_coll.append([prec_svm])
    reca_coll.append([reca_svm])
    f1_coll.append([f1_svm])
    acc_coll.append([acc_svm])


# Baseline standard classification

prec_coll, reca_coll, f1_coll, acc_coll = \
    np.array(prec_coll), np.array(reca_coll), np.array(f1_coll), np.array(acc_coll)

print(f'Average metrics over {iteration_count} iterations')
for index, method in enumerate(occ_methods):
    prec = f'{np.mean(prec_coll[:, index]).round(3)} \u00B1 {np.std(prec_coll[:, index]).round(3)}'
    reca = f'{np.mean(reca_coll[:, index]).round(3)} \u00B1 {np.std(reca_coll[:, index]).round(3)}'
    f1 = f'{np.mean(f1_coll[:, index]).round(3)} \u00B1 {np.std(f1_coll[:, index]).round(3)}'
    acc = f'{np.mean(acc_coll[:, index]).round(3)} \u00B1 {np.std(acc_coll[:, index]).round(3)}'

    print(tabulate([[method, prec, reca, f1, acc]], headers=['Method', 'Precision', 'Recall', 'F1 score', 'Accuracy']))

