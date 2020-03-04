from baselines.usv_baselines import svm_oneclass, elliptic_envelope, iso_forest, local_outlier_factor
from utils.run_models import run_usv_classification


def build_unsupervised_baselines(x_train, x_test, y_test):

    def evaluate_model(clf, lists, label):
        prec, reca, f1, auc = run_usv_classification(x_test, y_test, clf, 'fraud-prediction')
        lists['prec_list'].append(prec)
        lists['reca_list'].append(reca)
        lists['f1_list'].append(f1)
        lists['auc_list'].append(auc)
        lists['method_list'].append(label)
        return lists

    results = {
        'prec_list': list(),
        'reca_list': list(),
        'f1_list': list(),
        'auc_list': list(),
        'method_list': list()
    }

    # OC-SVM
    results = evaluate_model(svm_oneclass(x_train), results, 'OC-SVM')

    # Elliptic Envelope
    results = evaluate_model(elliptic_envelope(x_train), results, 'Elliptic Envelope')

    # Isolation Forest
    results = evaluate_model(iso_forest(x_train), results, 'Isolation Forest')

    # kNN Local Outlier Factor
    results = evaluate_model(local_outlier_factor(x_train), results, 'kNN Local Outlier Factor')

    return results['prec_list'], results['reca_list'], results['f1_list'], results['auc_list'], results['method_list']
