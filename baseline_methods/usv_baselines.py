from sklearn import covariance, svm, ensemble
from sklearn.neighbors import LocalOutlierFactor

from baseline_methods.utils import execute_decision_function


def svm_oneclass(train_test_split, label, result_list):
    clf = svm.OneClassSVM()

    return execute_decision_function(clf, train_test_split, label, result_list, unsupervised=True)


def elliptic_envelope(train_test_split, label, result_list):
    clf = covariance.EllipticEnvelope(support_fraction=1)

    return execute_decision_function(clf, train_test_split, label, result_list, unsupervised=True)


def iso_forest(train_test_split, label, result_list):
    x_train = train_test_split['x_train']
    clf = ensemble.IsolationForest(max_samples=x_train.shape[0], random_state=None)

    return execute_decision_function(clf, train_test_split, label, result_list, unsupervised=True)


def local_outlier_factor(train_test_split, label, result_list):
    clf = LocalOutlierFactor(novelty=True)

    return execute_decision_function(clf, train_test_split, label, result_list, unsupervised=True)
