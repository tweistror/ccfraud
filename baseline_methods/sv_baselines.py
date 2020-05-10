from sklearn import tree, ensemble, neighbors, svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier

import xgboost as xgb
from sklearn.neural_network import MLPClassifier

from baseline_methods.utils import execute_decision_function, execute_predict_proba


def svm_svc(train_test_split, label, result_list):
    clf = svm.SVC(kernel='rbf')

    return execute_decision_function(clf, train_test_split, label, result_list)


def knn(train_test_split, label, result_list):
    clf = neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance')

    return execute_predict_proba(clf, train_test_split, label, result_list)


def decision_tree(train_test_split, label, result_list):
    clf = tree.DecisionTreeClassifier()

    return execute_predict_proba(clf, train_test_split, label, result_list)


def random_forest(train_test_split, label, result_list):
    clf = ensemble.RandomForestClassifier()

    return execute_predict_proba(clf, train_test_split, label, result_list)


def svm_linearsvc(train_test_split, label, result_list):
    clf = svm.LinearSVC()

    return execute_decision_function(clf, train_test_split, label, result_list)


def gnb(train_test_split, label, result_list):
    clf = GaussianNB()

    return execute_predict_proba(clf, train_test_split, label, result_list)


def logistic_regression(train_test_split, label, result_list):
    clf = LogisticRegression(max_iter=1000)

    return execute_decision_function(clf, train_test_split, label, result_list)


def xgboost(train_test_split, label, result_list):
    clf = xgb.XGBClassifier(max_depth=10)

    return execute_predict_proba(clf, train_test_split, label, result_list)


def sgd(train_test_split, label, result_list):
    clf = SGDClassifier()

    return execute_decision_function(clf, train_test_split, label, result_list)


def gaussian_process(train_test_split, label, result_list):
    clf = GaussianProcessClassifier()

    return execute_predict_proba(clf, train_test_split, label, result_list)


def adaboost(train_test_split, label, result_list):
    clf = AdaBoostClassifier()

    return execute_decision_function(clf, train_test_split, label, result_list)


def mlp(train_test_split, label, result_list):
    clf = MLPClassifier()

    return execute_predict_proba(clf, train_test_split, label, result_list)
