from baseline_methods.usv_baselines import svm_oneclass, elliptic_envelope, iso_forest, local_outlier_factor
from baseline_methods.utils import binarize_usv_test_labels


def build_unsupervised_baselines(x_train, x_test, y_test):
    results = {
        'prec_list': list(),
        'reca_list': list(),
        'f1_list': list(),
        'acc_list': list(),
        'pr_auc_list': list(),
        'roc_auc_list': list(),
        'method_list': list(),
    }

    train_test_split = {
        'x_train': x_train,
        'y_train': [],
        'x_test': x_test,
        'y_test': binarize_usv_test_labels(y_test),
    }

    # OC-SVM
    results = svm_oneclass(train_test_split, 'OC-SVM', results)

    # Elliptic Envelope
    results = elliptic_envelope(train_test_split, 'Elliptic Envelope', results)

    # Isolation Forest
    results = iso_forest(train_test_split, 'Isolation Forest', results)

    # kNN Local Outlier Factor
    results = local_outlier_factor(train_test_split, 'kNN Local Outlier Factor', results)

    return results
