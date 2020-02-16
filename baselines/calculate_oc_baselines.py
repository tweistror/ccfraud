from baselines.oc_baselines import svm_oneclass, elliptic_envelope, iso_forest, local_outlier_factor
from utils.run_models import run_one_svm


def build_oc_baselines(x_train, x_test, y_test, train_size, test_negative_samples):

    def evaluate_model(clf, lists):
        prec, reca, f1, acc = run_one_svm(x_test, y_test, clf, test_negative_samples, 'fraud-prediction')
        lists['prec_list'].append(prec)
        lists['reca_list'].append(reca)
        lists['f1_list'].append(f1)
        lists['acc_list'].append(acc)
        return lists

    results = {
        'prec_list': list(),
        'reca_list': list(),
        'f1_list': list(),
        'acc_list': list(),
    }

    # OC-SVM
    results = evaluate_model(svm_oneclass(x_train[0:train_size]), results)

    # Elliptic Envelope
    results = evaluate_model(elliptic_envelope(x_train[0:train_size]),  results)

    # Isolation Forest
    results = evaluate_model(iso_forest(x_train[0:train_size]), results)

    # kNN Local Outlier Factor
    results = evaluate_model(local_outlier_factor(x_train[0:train_size]), results)

    return results['prec_list'], results['reca_list'], results['f1_list'], results['acc_list']
