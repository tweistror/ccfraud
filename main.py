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
    x_ben = sample_shuffle(x_ben)[0:sample_size]
elif dataset_string == "ccfraud":
    x_ben, x_fraud = get_data_ccfraud("ccfraud.csv")
    x_ben = sample_shuffle(x_ben)[0:sample_size]
elif dataset_string == "ieee":
    x_ben, x_fraud = get_data_ieee("ieee.csv")
    x_ben = sample_shuffle(x_ben)[0:sample_size]


column_count = x_ben.shape[1]
train_test_ratio = 0.75

iteration_count = 10

svm_prec_list = list()
svm_reca_list = list()
svm_f1_list = list()
svm_acc_list = list()

for i in range(iteration_count):
    if dataset_string == "paysim":
        print(dataset_string)
    elif dataset_string == "ccfraud":
        x_train, x_test, y_train, y_test = sample_data_for_occ(x_ben, x_fraud, train_test_ratio,
                                                               dataset_string)
    elif dataset_string == "ieee":
        print(dataset_string)

    # OC-SVM
    clf = svm_oneclass(x_train[0:sample_size])
    prec_svm, reca_svm, f1_svm, acc_svm = run_one_svm(x_test, y_test, clf, 'fraud-prediction')


    svm_prec_list.append(prec_svm)
    svm_reca_list.append(reca_svm)
    svm_f1_list.append(f1_svm)
    svm_acc_list.append(acc_svm)


prec_avg = (np.sum(svm_prec_list) / len(svm_prec_list)).round(4)
reca_avg = (np.sum(svm_reca_list) / len(svm_reca_list)).round(4)
f1_avg = (np.sum(svm_f1_list) / len(svm_f1_list)).round(4)
acc_avg = (np.sum(svm_acc_list) / len(svm_acc_list)).round(4)

print(f'Average metrics over {iteration_count} iterations')
print(tabulate([['OC-SVM', prec_avg, reca_avg, f1_avg, acc_avg]], headers=['Method', 'Precision', 'Recall', 'F1 score',
                                                                           'Accuracy']))


