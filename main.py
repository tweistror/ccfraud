import argparse
from argparse import RawTextHelpFormatter
import numpy as np
from tabulate import tabulate
from datetime import datetime

from baselines.calculate_sv_baselines import build_supervised_baselines
from baselines.calculate_usv_baselines import build_unsupervised_baselines
from utils.list_operations import sample_shuffle
from utils.load_data import get_data_paysim, get_data_ccfraud, get_data_ieee, get_parameters
from utils.sample_data import sample_data_for_unsupervised_baselines, sample_data_for_supervised_baselines

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
positive_samples = 10000


# Load data
if dataset_string == "paysim":
    x_ben, x_fraud = get_data_paysim("paysim.csv", positive_samples=positive_samples, verbosity=verbosity)
    x_ben = sample_shuffle(x_ben)
elif dataset_string == "ccfraud":
    x_ben, x_fraud = get_data_ccfraud("ccfraud.csv", positive_samples=positive_samples, verbosity=verbosity)
    x_ben = sample_shuffle(x_ben)
elif dataset_string == "ieee":
    x_ben, x_fraud = get_data_ieee("ieee_transaction.csv", "ieee_identity.csv", positive_samples=positive_samples, verbosity=verbosity)
    x_ben = sample_shuffle(x_ben)
    x_fraud = sample_shuffle(x_fraud[0:2000])

# Set parameters
usv_train, sv_train, sv_train_fraud, test_fraud = get_parameters(dataset_string)
iteration_count = 10

# Initialize collections for evaluation results
prec_coll = list()
reca_coll = list()
f1_coll = list()
acc_coll = list()
method_list = list()

start_time_complete = datetime.now()
if verbosity > 0:
    print(f'Start {iteration_count} iterations')

for i in range(iteration_count):
    start_time = datetime.now()
    if verbosity > 1:
        print(f'Starting iteration #{i+1}')

    # Sample data for unsupervised learning baselines
    x_train, x_test, y_train, y_test = sample_data_for_unsupervised_baselines(x_ben, x_fraud)
    # Execute unsupervised learning baselines
    prec_usv_list, reca_usv_list, f1_usv_list, acc_usv_list, method_usv_list = \
        build_unsupervised_baselines(x_train, x_test, y_test, usv_train, test_fraud)

    # Some verbosity output
    if verbosity > 1:
        print(f'Iteration #{i+1} unsupervised finished, supervised coming up')

    # Sample data for supervised learning baselines
    x_train, x_test, y_train, y_test = \
        sample_data_for_supervised_baselines(x_ben, x_fraud, sv_train, sv_train_fraud)
    # Execute supervised learning baselines
    prec_sv_list, reca_sv_list, f1_sv_list, acc_sv_list, method_sv_list = \
        build_supervised_baselines(x_train, y_train, x_test, y_test, test_fraud)

    # Add metrics for all methods to collections
    prec_coll.append(prec_usv_list + prec_sv_list)
    reca_coll.append(reca_usv_list + reca_sv_list)
    f1_coll.append(f1_usv_list + f1_sv_list)
    acc_coll.append(acc_usv_list + acc_sv_list)

    if i == 0:
        method_list = method_usv_list + method_sv_list

    if verbosity > 0:
        time_required = str(datetime.now() - start_time)
        print(f'Iteration #{i+1} finished in {time_required}')


if verbosity > 1:
    time_required = str(datetime.now() - start_time_complete)
    print(f'All {iteration_count} iterations finished in {time_required}')

prec_coll, reca_coll, f1_coll, acc_coll, method_list = \
    np.array(prec_coll), np.array(reca_coll), np.array(f1_coll), np.array(acc_coll), np.array(method_list)

print(f'Average metrics over {iteration_count} iterations')

results = list()

for index, method in enumerate(method_list):
    if index == 0:
        results.append(['Unsupervised Learning Methods'])

    if index == len(method_usv_list):
        results.append(['Supervised Learning Methods'])

    prec = f'{np.mean(prec_coll[:, index]).round(3)} \u00B1 {np.std(prec_coll[:, index]).round(3)}'
    reca = f'{np.mean(reca_coll[:, index]).round(3)} \u00B1 {np.std(reca_coll[:, index]).round(3)}'
    f1 = f'{np.mean(f1_coll[:, index]).round(3)} \u00B1 {np.std(f1_coll[:, index]).round(3)}'
    acc = f'{np.mean(acc_coll[:, index]).round(3)} \u00B1 {np.std(acc_coll[:, index]).round(3)}'
    results.append([method, prec, reca, f1, acc])

print(tabulate(results, headers=['Method', 'Precision', 'Recall', 'F1 score', 'Accuracy']))

