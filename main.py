import argparse
from argparse import RawTextHelpFormatter
import numpy as np
from datetime import datetime

from advanced.oc_gan.oc_gan import execute_oc_gan
from baselines.calculate_sv_baselines import build_supervised_baselines
from baselines.calculate_usv_baselines import build_unsupervised_baselines
from utils.load_data import get_data_paysim, get_data_ccfraud, get_data_ieee, get_parameters
from utils.printing import print_results
from utils.sample_data import sample_paysim, sample_ccfraud, sample_ieee

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
parser.add_argument("--iterations", default="10", help="Specify number of iterations each method is executed")
parser.add_argument("--cv", help="Specify number of cross validation splits")

args = parser.parse_args()
dataset_string = args.dataset
verbosity = int(args.v)
method = args.method
baselines = args.baselines
iteration_count = int(args.iterations)
cross_validation_count = 0 if args.cv is None else int(args.cv)

# Set parameters
usv_train, sv_train, sv_train_fraud, test_fraud = get_parameters(dataset_string, cross_validation_count)

skip_ieee_processing = True

if dataset_string == "paysim":
    x_ben, x_fraud = get_data_paysim("paysim.csv", verbosity=verbosity)
elif dataset_string == "ccfraud":
    x_ben, x_fraud = get_data_ccfraud("ccfraud.csv", verbosity=verbosity)
elif dataset_string == "ieee":
    x_ben, x_fraud = get_data_ieee("ieee_transaction.csv", "ieee_identity.csv", verbosity=verbosity,
                                   skip=skip_ieee_processing)

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

    if dataset_string == "paysim":
        x_usv_train, x_sv_train, y_sv_train, x_test, y_test = sample_paysim(x_ben, x_fraud, usv_train, sv_train,
                                                                            sv_train_fraud, test_fraud,
                                                                            cross_validation_count)
    elif dataset_string == "ccfraud":
        x_usv_train, x_sv_train, y_sv_train, x_test, y_test = sample_ccfraud(x_ben, x_fraud, usv_train, sv_train,
                                                                             sv_train_fraud, test_fraud,
                                                                             cross_validation_count)
    elif dataset_string == "ieee":
        x_usv_train, x_sv_train, y_sv_train, x_test, y_test = sample_ieee(x_ben, x_fraud, usv_train, sv_train,
                                                                          sv_train_fraud, test_fraud)

    if verbosity > 1:
        print(f'Starting iteration #{i + 1}')

    if method == 'oc-gan':
        prec, reca, f1, acc, method_name = execute_oc_gan(dataset_string, x_usv_train, x_test, y_test)
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
        # Execute unsupervised learning baselines
        if cross_validation_count > 0:
            temp_prec_list = list()
            temp_reca_list = list()
            temp_f1_list = list()
            temp_acc_list = list()

            for index in range(cross_validation_count):
                prec_list_1, reca_list_1, f1_list_1, acc_list_1, method_usv_list = \
                    build_unsupervised_baselines(x_usv_train[index * usv_train:(index + 1) * usv_train],
                                                 x_test, y_test)
                temp_prec_list.append(prec_list_1)
                temp_reca_list.append(reca_list_1)
                temp_f1_list.append(f1_list_1)
                temp_acc_list.append(acc_list_1)

            temp_prec_list, temp_reca_list, temp_f1_list, temp_acc_list = \
                np.array(temp_prec_list), np.array(temp_reca_list), np.array(temp_f1_list), np.array(temp_acc_list)

            prec_usv_list = list()
            reca_usv_list = list()
            f1_usv_list = list()
            acc_usv_list = list()
            for index, method in enumerate(method_usv_list):
                prec = np.mean(temp_prec_list[:, index]).round(3)
                reca = np.mean(temp_reca_list[:, index]).round(3)
                f1 = np.mean(temp_f1_list[:, index]).round(3)
                acc = np.mean(temp_acc_list[:, index]).round(3)
                prec_usv_list.append(prec)
                reca_usv_list.append(reca)
                f1_usv_list.append(f1)
                acc_usv_list.append(acc)

        else:
            prec_usv_list, reca_usv_list, f1_usv_list, acc_usv_list, method_usv_list = \
                build_unsupervised_baselines(x_usv_train, x_test, y_test)

        # Add metrics to collections
        prec_list = prec_list + prec_usv_list
        reca_list = reca_list + reca_usv_list
        f1_list = f1_list + f1_usv_list
        acc_list = acc_list + acc_usv_list

        # Some verbosity output
        if verbosity > 1:
            print(f'Unsupervised: Iteration #{i + 1} finished')

    if baselines == 'sv' or baselines == 'both':
        # Execute supervised learning baselines
        if cross_validation_count > 0:
            temp_prec_list = list()
            temp_reca_list = list()
            temp_f1_list = list()
            temp_acc_list = list()

            for index in range(cross_validation_count):
                prec_list_1, reca_list_1, f1_list_1, acc_list_1, method_sv_list = \
                    build_supervised_baselines(x_sv_train[index * sv_train:(index + 1) * sv_train],
                                               y_sv_train[index * sv_train:(index + 1) * sv_train], x_test, y_test)
                temp_prec_list.append(prec_list_1)
                temp_reca_list.append(reca_list_1)
                temp_f1_list.append(f1_list_1)
                temp_acc_list.append(acc_list_1)

            temp_prec_list, temp_reca_list, temp_f1_list, temp_acc_list = \
                np.array(temp_prec_list), np.array(temp_reca_list), np.array(temp_f1_list), np.array(temp_acc_list)

            prec_sv_list = list()
            reca_sv_list = list()
            f1_sv_list = list()
            acc_sv_list = list()
            for index, method in enumerate(method_sv_list):
                prec = np.mean(temp_prec_list[:, index]).round(3)
                reca = np.mean(temp_reca_list[:, index]).round(3)
                f1 = np.mean(temp_f1_list[:, index]).round(3)
                acc = np.mean(temp_acc_list[:, index]).round(3)
                prec_sv_list.append(prec)
                reca_sv_list.append(reca)
                f1_sv_list.append(f1)
                acc_sv_list.append(acc)

        else:
            prec_sv_list, reca_sv_list, f1_sv_list, acc_sv_list, method_sv_list = \
                build_supervised_baselines(x_sv_train, y_sv_train, x_test, y_test)

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
