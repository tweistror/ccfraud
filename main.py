import argparse
from argparse import RawTextHelpFormatter
import numpy as np
from datetime import datetime

from advanced_methods.AE.autoencoder import Autoencoder
from advanced_methods.RBM.rbm import RBM
from advanced_methods.VAE.vae import VAE
from advanced_methods.OC_GAN.oc_gan import execute_oc_gan
from baseline_methods.evaluate_sv_baselines import build_supervised_baselines
from baseline_methods.evaluate_usv_baselines import build_unsupervised_baselines
from utils.crossvalidator import Crossvalidator
from utils.load_data import get_data_paysim, get_data_ccfraud, get_data_ieee, get_parameters, get_data_paysim_custom
from utils.printing import print_results
from utils.split_preprocess_data import execute_smote, split_and_preprocess_data

datasets = ["paysim", "paysim_custom", "ccfraud", "ieee"]
methods = ["all", "oc-gan", "oc-gan-ae", "ae", "rbm", "vae"]
baselines = ["both", "usv", "sv"]

parser = argparse.ArgumentParser(description='Tool for testing various machine learning methods on different datasets',
                                 formatter_class=RawTextHelpFormatter)
parser.add_argument("--dataset", required=True, choices=datasets, help="Desired dataset for evaluation")
parser.add_argument("--method", choices=methods,
                    help="Method for evaluation (no specification will result in no evaluation of any advanced method, "
                         "all with execute all advanced methods)")
parser.add_argument("--baselines", choices=baselines, default="both",
                    help="Baselines for evaluation (default is both)")
parser.add_argument("--v", choices=['0', '1', '2'], default=0, help="Verbosity level (0 = just end results, 1 = "
                                                                    "some timing information, "
                                                                    "2 = more timing information)")
parser.add_argument("--iterations", default="10", help="Desired count the specified methods are executed and evaluated")
parser.add_argument("--cv", help="Activate crossvalidation with the desired count of train/test-splits")
parser.add_argument("--oversampling", choices=['y', 'n'], default='n', help="Use oversampling (SMOTE) or not")

args = parser.parse_args()
dataset_string = args.dataset
verbosity = int(args.v)
method = args.method
baseline = args.baselines
iteration_count = int(args.iterations)
use_oversampling = True if args.oversampling == 'y' else False
cross_validation_count = 1 if args.cv is None else int(args.cv)

# Set parameters
usv_train, sv_train, sv_train_fraud, test_fraud, test_benign = get_parameters(dataset_string, cross_validation_count)

skip_ieee_processing = True

if dataset_string == "paysim":
    x_ben, x_fraud = get_data_paysim("paysim.csv", verbosity=verbosity)
if dataset_string == "paysim_custom":
    x_ben, x_fraud = get_data_paysim_custom("paysim_custom.csv", verbosity=verbosity)
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

    x_usv_train, x_sv_train, y_sv_train, x_test, y_test = split_and_preprocess_data(dataset_string, x_ben, x_fraud,
                                                                                    usv_train, sv_train, sv_train_fraud,
                                                                                    test_fraud, test_benign,
                                                                                    cross_validation_count)

    # Over/undersampling
    if use_oversampling is True:
        x_sv_train, y_sv_train = execute_smote(x_sv_train, y_sv_train)

    if verbosity > 1:
        print(f'Starting iteration #{i + 1}')

    if method == 'all' or method == 'oc-gan':
        prec, reca, f1, auc, method_name = execute_oc_gan(dataset_string, x_usv_train, x_test[:test_benign],
                                                          x_test[test_benign:], test_benign, autoencoding=False)
        prec_list = prec_list + [prec]
        reca_list = reca_list + [reca]
        f1_list = f1_list + [f1]
        auc_list = auc_list + [auc]
        if i == 0:
            method_special_list = method_special_list + [method_name]

    if method == 'all' or method == 'oc-gan-ae':
        prec, reca, f1, auc, method_name = execute_oc_gan(dataset_string, x_usv_train, x_test[:test_benign],
                                                          x_test[test_benign:], test_benign, autoencoding=True)
        prec_list = prec_list + [prec]
        reca_list = reca_list + [reca]
        f1_list = f1_list + [f1]
        auc_list = auc_list + [auc]
        if i == 0:
            method_special_list = method_special_list + [method_name]

    if method == 'all' or method == 'ae':
        ae_model = Autoencoder(x_usv_train, dataset_string)
        ae_model.set_parameters()
        ae_model.build()
        prec, reca, f1, auc, method_name = ae_model.predict(x_test, y_test)

        prec_list = prec_list + [prec]
        reca_list = reca_list + [reca]
        f1_list = f1_list + [f1]
        auc_list = auc_list + [auc]
        if i == 0:
            method_special_list = method_special_list + [method_name]

    if method == 'all' or method == 'rbm':
        rbm_model = RBM(x_usv_train.shape[1], 10, visible_unit_type='gauss', gibbs_sampling_steps=4,
                        learning_rate=0.001, momentum=0.95, batch_size=512, num_epochs=10, verbose=0)
        prec, reca, f1, auc, method_name = rbm_model.execute(x_usv_train, x_test, y_test)

        prec_list = prec_list + [prec]
        reca_list = reca_list + [reca]
        f1_list = f1_list + [f1]
        auc_list = auc_list + [auc]
        if i == 0:
            method_special_list = method_special_list + [method_name]

    if method == 'all' or method == 'vae':
        vae_model = VAE(x_usv_train, dataset_string)
        vae_model.set_parameters()
        vae_model.build()
        prec, reca, f1, auc, method_name = vae_model.predict(x_test, y_test)

        prec_list = prec_list + [prec]
        reca_list = reca_list + [reca]
        f1_list = f1_list + [f1]
        auc_list = auc_list + [auc]
        if i == 0:
            method_special_list = method_special_list + [method_name]

    # Some verbosity output
    if verbosity > 1:
        print(f'Special methods: Iteration #{i + 1} finished')

    if baseline == 'usv' or baseline == 'both':
        # Execute unsupervised learning baseline methods
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

    if baseline == 'sv' or baseline == 'both':
        # Execute supervised learning baseline methods
        if cross_validation_count > 1:
            cv = Crossvalidator(cross_validation_count, x_sv_train, y_sv_train)
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
