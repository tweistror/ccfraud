import argparse
from argparse import RawTextHelpFormatter
import numpy as np
from datetime import datetime

from advanced.oc_gan.oc_gan import execute_oc_gan
from baselines.calculate_sv_baselines import build_supervised_baselines
from baselines.calculate_usv_baselines import build_unsupervised_baselines
from utils.list_operations import sample_shuffle
from utils.load_data import get_data_paysim, get_data_ccfraud, get_data_ieee, get_parameters
from utils.printing import print_results
from utils.sample_data import sample_data_for_unsupervised_baselines, sample_data_for_supervised_baselines

datasets = ["paysim", "ccfraud", "ieee"]

parser = argparse.ArgumentParser(description='Tool for testing various machine learning methods on different datasets',
                                 formatter_class=RawTextHelpFormatter)
parser.add_argument("--dataset", required=True, choices=datasets, help="Dataset")
parser.add_argument("--method", choices=["oc-gan"],
                    help="Machine learning method used for classification")
parser.add_argument("--baselines", choices=["usv", "sv", "both"],
                    help="Execute baselines or not")
parser.add_argument("--mode", choices=["baseline", "solo"], help='''Execution mode: 
`baseline` for comparison to other baseline methods
`solo` for executing the chosen method only''')
parser.add_argument("--v", choices=['0', '1', '2'], default=0, help="Specify verbosity")
parser.add_argument("--iterations", default=10, help="Specify number of iterations each method is executed")

args = parser.parse_args()
dataset_string = args.dataset
verbosity = int(args.v)
method = args.method
baselines = args.baselines
iteration_count = int(args.iterations)

# Specify positive samples to load
positive_samples = 20000

if dataset_string == "paysim":
    x_ben, x_fraud = get_data_paysim("paysim.csv", positive_samples=positive_samples, verbosity=verbosity)
    x_ben = sample_shuffle(x_ben)
elif dataset_string == "ccfraud":
    x_ben, x_fraud = get_data_ccfraud("ccfraud.csv", positive_samples=positive_samples, verbosity=verbosity)
    x_ben = sample_shuffle(x_ben)
elif dataset_string == "ieee":
    x_ben, x_fraud = get_data_ieee("ieee_transaction.csv", "ieee_identity.csv", positive_samples=positive_samples,
                                   verbosity=verbosity)
    x_ben = sample_shuffle(x_ben)
    x_fraud = sample_shuffle(x_fraud[0:2000])

# Set parameters
usv_train, sv_train, sv_train_fraud, test_fraud = get_parameters(dataset_string)

# Initialize collections for evaluation results
prec_coll = list()
reca_coll = list()
f1_coll = list()
acc_coll = list()
method_list = list()

method_special_list = list()
method_usv_list = list()
method_sv_list = list()

start_time_complete = datetime.now()
if verbosity > 0:
    print(f'Start {iteration_count} iterations')

for i in range(iteration_count):
    start_time = datetime.now()

    prec_list = list()
    reca_list = list()
    f1_list = list()
    acc_list = list()

    if verbosity > 1:
        print(f'Starting iteration #{i + 1}')

    if method == 'oc-gan':
        prec, reca, f1, acc, method_name = execute_oc_gan(dataset_string, x_ben, x_fraud, usv_train, test_fraud)
        prec_list = prec_list + [prec]
        reca_list = reca_list + [reca]
        f1_list = f1_list + [f1]
        acc_list = acc_list + [acc]
        if i == 0:
            method_special_list = method_special_list + [method_name]

        # Some verbosity output
        if verbosity > 1:
            print(f'Special methods: Iteration #{i + 1} finished')

    if baselines == 'usv' or baselines == 'both':
        # Sample data for unsupervised learning baselines
        x_train, x_test, y_test = sample_data_for_unsupervised_baselines(x_ben, x_fraud, usv_train)
        # Execute unsupervised learning baselines
        prec_usv_list, reca_usv_list, f1_usv_list, acc_usv_list, method_usv_list = \
            build_unsupervised_baselines(x_train, x_test, y_test, test_fraud)

        # Add metrics to collections
        prec_list = prec_list + prec_usv_list
        reca_list = reca_list + reca_usv_list
        f1_list = f1_list + f1_usv_list
        acc_list = acc_list + acc_usv_list

        # Some verbosity output
        if verbosity > 1:
            print(f'Unsupervised: Iteration #{i + 1} finished')

    if baselines == 'sv' or baselines == 'both':
        # Sample data for supervised learning baselines
        x_train, x_test, y_train, y_test = \
            sample_data_for_supervised_baselines(x_ben, x_fraud, sv_train, sv_train_fraud)
        # Execute supervised learning baselines
        prec_sv_list, reca_sv_list, f1_sv_list, acc_sv_list, method_sv_list = \
            build_supervised_baselines(x_train, y_train, x_test, y_test, test_fraud)

        # Add metrics to collections
        prec_list = prec_list + prec_sv_list
        reca_list = reca_list + reca_sv_list
        f1_list = f1_list + f1_sv_list
        acc_list = acc_list + acc_sv_list

        # Some verbosity output
        if verbosity > 1:
            print(f'Supervised: Iteration #{i + 1} finished')

    prec_coll.append(prec_list)
    reca_coll.append(reca_list)
    f1_coll.append(f1_list)
    acc_coll.append(acc_list)

    if i == 0:
        method_list = method_special_list + method_usv_list + method_sv_list

    if verbosity > 0:
        time_required = str(datetime.now() - start_time)
        print(f'Iteration #{i + 1} finished in {time_required}')

if verbosity > 1:
    time_required = str(datetime.now() - start_time_complete)
    print(f'All {iteration_count} iterations finished in {time_required}')

prec_coll, reca_coll, f1_coll, acc_coll, method_list = \
    np.array(prec_coll), np.array(reca_coll), np.array(f1_coll), np.array(acc_coll), np.array(method_list)

print_results(method_list, dataset_string, iteration_count,
              len(method_special_list), len(method_usv_list), prec_coll, reca_coll, f1_coll, acc_coll)
