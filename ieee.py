import numpy as np
from tabulate import tabulate

from baselines.calculate_sv_baselines import build_supervised_baselines
from baselines.calculate_usv_baselines import build_unsupervised_baselines
from utils.list_operations import sample_shuffle
from utils.load_data import get_data_ieee
from utils.sample_data import sample_data_for_unsupervised_baselines, sample_data_for_supervised_baselines

positive_samples = 10000

unsupervised_train_size = 500
supervised_train_size = 2000
supervised_train_negative_samples = 50
test_negative_samples = 500

# x_ben, x_fraud = get_data_ieee("ieee_transaction.csv", "ieee_identity.csv", positive_samples=positive_samples,
#                                verbosity=1)
# x_ben = sample_shuffle(x_ben)
# x_fraud = sample_shuffle(x_fraud[0:2000])
# np.save('x_ben.npy', x_ben)
# np.save('x_fraud.npy', x_fraud)

x_ben = np.load('./debug/ieee/x_ben.npy')
x_fraud = np.load('./debug/ieee/x_fraud.npy')

prec_coll = list()
reca_coll = list()
f1_coll = list()
acc_coll = list()

# Sample data for unsupervised learning baselines
x_train, x_test, y_train, y_test = sample_data_for_unsupervised_baselines(x_ben, x_fraud)
# Execute unsupervised learning baselines
prec_oc_list, reca_oc_list, f1_oc_list, acc_oc_list = build_unsupervised_baselines(x_train, x_test, y_test,
                                                                                   unsupervised_train_size,
                                                                                   test_negative_samples)

# Sample data for supervised learning baselines
x_train, x_test, y_train, y_test = \
    sample_data_for_supervised_baselines(x_ben, x_fraud, supervised_train_size, supervised_train_negative_samples)
# Execute supervised learning baselines
prec_list, reca_list, f1_list, acc_list = build_supervised_baselines(x_train, y_train, x_test, y_test,
                                                                     test_negative_samples)

# Add metrics for all methods to collections
prec_coll.append(prec_oc_list + prec_list)
reca_coll.append(reca_oc_list + reca_list)
f1_coll.append(f1_oc_list + f1_list)
acc_coll.append(acc_oc_list + acc_list)

occ_methods = ['OC-SVM', 'Elliptic Envelope', 'Isolation Forest', 'kNN Local Outlier Factor']
baseline_methods = ['SVM SVC', 'kNN', 'Decision Tree', 'Random Forest', 'SVM Linear SVC', 'Gaussian NB',
                    'Logistic Regression', 'XG Boost', 'SGD', 'Gaussian Process', 'Decision Tree', 'Adaboost']


prec_coll, reca_coll, f1_coll, acc_coll = \
    np.array(prec_coll), np.array(reca_coll), np.array(f1_coll), np.array(acc_coll)


methods = occ_methods + baseline_methods

results = list()

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

