from baseline_methods.usv_baselines import USV_Baselines
from baseline_methods.utils import binarize_usv_test_labels


def build_unsupervised_baselines(x_train, x_test, y_test, image_creator):
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

    usv_baselines_class = USV_Baselines(image_creator, train_test_split)

    # OC-SVM
    results = usv_baselines_class.svm_oneclass('OC-SVM', results)

    # Elliptic Envelope
    results = usv_baselines_class.elliptic_envelope('Elliptic Envelope', results)

    # Isolation Forest
    results = usv_baselines_class.iso_forest('Isolation Forest', results)

    # kNN Local Outlier Factor
    results = usv_baselines_class.local_outlier_factor('kNN Local Outlier Factor', results)

    return results
