import argparse
from argparse import RawTextHelpFormatter
import numpy as np
from datetime import datetime

from advanced.AE.autoencoder import Autoencoder
from advanced.oc_gan.oc_gan import execute_oc_gan
from baselines.calculate_sv_baselines import build_supervised_baselines
from baselines.calculate_usv_baselines import build_unsupervised_baselines
from utils.crossvalidation import Crossvalidator
from utils.load_data import get_data_paysim, get_data_ccfraud, get_data_ieee, get_parameters
from utils.printing import print_results
from utils.sample_data import sample_paysim, sample_ccfraud, sample_ieee, execute_nearmiss, execute_smote

datasets = ["paysim", "ccfraud", "ieee"]

parser = argparse.ArgumentParser(description='Tool for testing various machine learning methods on different datasets',
                                 formatter_class=RawTextHelpFormatter)
parser.add_argument("--dataset", required=True, choices=datasets, help="Dataset")
parser.add_argument("--method", choices=["all", "oc-gan", "oc-gan-ae", "usv-ae"],
                    help="Machine learning method used for classification")
parser.add_argument("--baselines", choices=["usv", "sv", "both"],
                    help="Execute baselines or not")
parser.add_argument("--v", choices=['0', '1', '2'], default=0, help="Specify verbosity")
parser.add_argument("--iterations", default="10", help="Specify number of iterations each method is executed")
parser.add_argument("--cv", help="Specify number of cross validation splits")
parser.add_argument("--oversampling", choices=['y', 'n'], default='n', help="Use oversampling (SMOTE) or not")
# TODO: Requirement: Oversampling only possible for oversampling `y`

args = parser.parse_args()
dataset_string = args.dataset
verbosity = int(args.v)
method = args.method
baselines = args.baselines
iteration_count = int(args.iterations)
use_oversampling = True if args.oversampling == 'y' else False
cross_validation_count = 1 if args.cv is None else int(args.cv)

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
auc_coll = list()
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
    auc_list = list()

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
                                                                          sv_train_fraud, test_fraud,
                                                                          cross_validation_count)
    # Use ccfraud as fallback
    else:
        x_usv_train, x_sv_train, y_sv_train, x_test, y_test = sample_ccfraud(x_ben, x_fraud, usv_train, sv_train,
                                                                             sv_train_fraud, test_fraud,
                                                                             cross_validation_count)

    # Over/undersampling
    if use_oversampling is True:
        x_sv_train, y_sv_train = execute_smote(x_sv_train, y_sv_train)

    # x_sv_train, y_sv_train = execute_nearmiss(x_sv_train, y_sv_train)

    # tsne_plot(x_sv_train, y_sv_train, "original.png")
    # ex_ae(x_usv_train, x_ben, x_fraud, x_test, y_test)

    # gan = GAN()
    # gan.train(epochs=30000, batch_size=32, sample_interval=200)

    if verbosity > 1:
        print(f'Starting iteration #{i + 1}')

    if method == 'all' or method == 'oc-gan':
        prec, reca, f1, auc, method_name = execute_oc_gan(dataset_string, x_usv_train, x_test[:-test_fraud],
                                                          x_test[-test_fraud:], test_fraud, autoencoding=False)
        prec_list = prec_list + [prec]
        reca_list = reca_list + [reca]
        f1_list = f1_list + [f1]
        auc_list = auc_list + [auc]
        if i == 0:
            method_special_list = method_special_list + [method_name]

    if method == 'all' or method == 'oc-gan-ae':
        prec, reca, f1, auc, method_name = execute_oc_gan(dataset_string, x_usv_train, x_test[:-test_fraud],
                                                          x_test[-test_fraud:], test_fraud, autoencoding=True)
        prec_list = prec_list + [prec]
        reca_list = reca_list + [reca]
        f1_list = f1_list + [f1]
        auc_list = auc_list + [auc]
        if i == 0:
            method_special_list = method_special_list + [method_name]

    if method == 'all' or method == 'usv_ae':
        ae_model = Autoencoder(dataset_string, x_usv_train, x_test, y_test)
        prec, reca, f1, auc, method_name = ae_model.execute_autoencoder()

        prec_list = prec_list + [prec]
        reca_list = reca_list + [reca]
        f1_list = f1_list + [f1]
        auc_list = auc_list + [auc]
        if i == 0:
            method_special_list = method_special_list + [method_name]

    # Some verbosity output
    if verbosity > 1:
        print(f'Special methods: Iteration #{i + 1} finished')

    if baselines == 'usv' or baselines == 'both':
        # Execute unsupervised learning baselines
        prec_usv_list, reca_usv_list, f1_usv_list, auc_usv_list, method_usv_list = \
            build_unsupervised_baselines(x_usv_train, x_test, y_test)

        # Add metrics to collections
        prec_list = prec_list + prec_usv_list
        reca_list = reca_list + reca_usv_list
        f1_list = f1_list + f1_usv_list
        auc_list = auc_list + auc_usv_list

        # Some verbosity output
        if verbosity > 1:
            print(f'Unsupervised: Iteration #{i + 1} finished')

    if baselines == 'sv' or baselines == 'both':
        # Execute supervised learning baselines
        if cross_validation_count > 1:
            cv = Crossvalidator(cross_validation_count, 'StratifiedKFold', x_sv_train, y_sv_train)
            prec_sv_list, reca_sv_list, f1_sv_list, auc_sv_list, method_sv_list = cv.execute_cv()
        else:
            prec_sv_list, reca_sv_list, f1_sv_list, auc_sv_list, method_sv_list = \
                build_supervised_baselines(x_sv_train, y_sv_train, x_test, y_test)

        # Add metrics to collections
        prec_list = prec_list + prec_sv_list
        reca_list = reca_list + reca_sv_list
        f1_list = f1_list + f1_sv_list
        auc_list = auc_list + auc_sv_list

        # Some verbosity output
        if verbosity > 1:
            print(f'Supervised: Iteration #{i + 1} finished')

    prec_coll.append(prec_list)
    reca_coll.append(reca_list)
    f1_coll.append(f1_list)
    auc_coll.append(auc_list)

    if i == 0:
        method_list = method_special_list + method_usv_list + method_sv_list

    if verbosity > 0:
        time_required = str(datetime.now() - start_time)
        print(f'Iteration #{i + 1} finished in {time_required}')

if verbosity > 1:
    time_required = str(datetime.now() - start_time_complete)
    print(f'All {iteration_count} iterations finished in {time_required}')

prec_coll, reca_coll, f1_coll, auc_coll, method_list = \
    np.array(prec_coll), np.array(reca_coll), np.array(f1_coll), np.array(auc_coll), np.array(method_list)

print_results(method_list, dataset_string, iteration_count,
              len(method_special_list), len(method_usv_list), prec_coll, reca_coll, f1_coll, auc_coll)
