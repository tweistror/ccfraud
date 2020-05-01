import numpy as np
from datetime import datetime

from advanced_methods.AE.autoencoder import Autoencoder
from advanced_methods.RBM.rbm import RBM
from advanced_methods.VAE.vae import VAE
from advanced_methods.OC_GAN.oc_gan import execute_oc_gan
from baseline_methods.evaluate_sv_baselines import build_supervised_baselines
from baseline_methods.evaluate_usv_baselines import build_unsupervised_baselines
from utils.crossvalidator import Crossvalidator
from utils.load_data import LoadData
from utils.parameters import Parameters
from utils.parser import Parser
from utils.printing import print_results
from utils.smote import execute_smote
from utils.split_preprocess_data import SplitPreprocessData

datasets = ["paysim", "paysim-custom", "ccfraud", "ieee", "nslkdd", "saperp-ek", "saperp-vk"]
methods = ["all", "oc-gan", "oc-gan-ae", "ae", "rbm", "vae"]
baselines = ["both", "usv", "sv", "none"]

parser = Parser(datasets, methods, baselines)

dataset_string, verbosity, seed, method, baseline, iteration_count, use_oversampling, cross_validation_count = \
    parser.get_args()

# Set parameters
parameter_class = Parameters(dataset_string)

usv_train, sv_train, sv_train_fraud, test_benign, test_fraud = \
    parameter_class.get_main_parameters(cross_validation_count)

x_ben, x_fraud = LoadData(dataset_string, parameter_class.get_path(), seed, verbosity).get_data()

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

    split_preprocess_class = SplitPreprocessData(dataset_string, seed, cross_validation_count, verbosity)
    x_usv_train, x_sv_train, y_sv_train, x_test, y_test = \
        split_preprocess_class.execute_split_preprocess(x_ben, x_fraud, usv_train, sv_train,
                                                        sv_train_fraud, test_fraud, test_benign)

    # Over/undersampling
    if use_oversampling is True:
        x_sv_train, y_sv_train = execute_smote(x_sv_train, y_sv_train, seed)

    if verbosity > 1:
        print(f'Starting iteration #{i + 1}')

    if method == 'all' or method == 'oc-gan':
        prec, reca, f1, acc, method_name = execute_oc_gan(x_usv_train, x_test[:test_benign],
                                                          x_test[test_benign:], test_benign,
                                                          parameter_class.get_oc_gan_parameters(),
                                                          autoencoding=False, verbosity=verbosity)
        prec_list = prec_list + [prec]
        reca_list = reca_list + [reca]
        f1_list = f1_list + [f1]
        acc_list = acc_list + [acc]
        if i == 0:
            method_special_list = method_special_list + [method_name]

    if method == 'all' or method == 'oc-gan-ae':
        prec, reca, f1, acc, method_name = execute_oc_gan(dataset_string, x_usv_train, x_test[:test_benign],
                                                          x_test[test_benign:], test_benign,
                                                          parameter_class.get_oc_gan_parameters(),
                                                          autoencoding=True, verbosity=verbosity)
        prec_list = prec_list + [prec]
        reca_list = reca_list + [reca]
        f1_list = f1_list + [f1]
        acc_list = acc_list + [acc]
        if i == 0:
            method_special_list = method_special_list + [method_name]

    if method == 'all' or method == 'ae':
        ae_model = Autoencoder(x_usv_train, dataset_string, verbosity=verbosity)
        ae_model.set_parameters(parameter_class.get_autoencoder_parameters())
        ae_model.build()
        prec, reca, f1, acc, method_name = ae_model.predict(x_test, y_test)

        prec_list = prec_list + [prec]
        reca_list = reca_list + [reca]
        f1_list = f1_list + [f1]
        acc_list = acc_list + [acc]
        if i == 0:
            method_special_list = method_special_list + [method_name]

    if method == 'all' or method == 'rbm':
        rbm_model = RBM(verbosity=verbosity)
        rbm_model.set_parameters(x_usv_train.shape[1], parameter_class.get_rbm_parameters())
        prec, reca, f1, acc, method_name = rbm_model.execute(x_usv_train, x_test, y_test)

        prec_list = prec_list + [prec]
        reca_list = reca_list + [reca]
        f1_list = f1_list + [f1]
        acc_list = acc_list + [acc]
        if i == 0:
            method_special_list = method_special_list + [method_name]

    if method == 'all' or method == 'vae':
        vae_model = VAE(x_usv_train, dataset_string, verbosity=verbosity)
        vae_model.set_parameters(parameter_class.get_vae_parameters())
        vae_model.build()
        prec, reca, f1, acc, method_name = vae_model.predict(x_test, y_test)

        prec_list = prec_list + [prec]
        reca_list = reca_list + [reca]
        f1_list = f1_list + [f1]
        acc_list = acc_list + [acc]
        if i == 0:
            method_special_list = method_special_list + [method_name]

    # Some verbosity output
    if verbosity > 1:
        print(f'Special methods: Iteration #{i + 1} finished')

    if baseline == 'usv' or baseline == 'both':
        # Execute unsupervised learning baseline methods
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

    if baseline == 'sv' or baseline == 'both':
        # Execute supervised learning baseline methods
        if cross_validation_count > 1:
            cv = Crossvalidator(cross_validation_count, x_sv_train, y_sv_train)
            prec_sv_list, reca_sv_list, f1_sv_list, acc_sv_list, method_sv_list = cv.execute_cv()
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
              len(method_special_list), len(method_usv_list), prec_coll, reca_coll, f1_coll, acc_coll, usv_train,
              sv_train, sv_train_fraud)
