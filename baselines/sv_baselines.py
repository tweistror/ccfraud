from sklearn import tree, ensemble, neighbors, svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier

import xgboost as xgb


def svm_svc(x_train, y_train):
    # TODO: Use different kernels?
    clf = svm.SVC(kernel='rbf')
    return clf.fit(x_train, y_train)


def knn(x_train, y_train):
    clf = neighbors.KNeighborsClassifier(n_neighbors=3)
    return clf.fit(x_train, y_train)


def decision_tree(x_train, y_train):
    clf = tree.DecisionTreeClassifier()
    return clf.fit(x_train, y_train)


def random_forest(x_train, y_train):
    clf = ensemble.RandomForestClassifier()
    return clf.fit(x_train, y_train)


def svm_linearsvc(x, y):
    clf = svm.LinearSVC()
    return clf.fit(x, y)


def gnb(x_train, y_train):
    clf = GaussianNB()
    return clf.fit(x_train, y_train)


def logistic_regression(x_train, y_train):
    clf = LogisticRegression(max_iter=1000)
    return clf.fit(x_train, y_train)


def xgboost(x_train, y_train):
    clf = xgb.XGBClassifier(max_depth=10)
    return clf.fit(x_train, y_train)


def sgd(x_train, y_train):
    clf = SGDClassifier()
    return clf.fit(x_train, y_train)


def gaussian_process(x_train, y_train):
    clf = GaussianProcessClassifier()
    return clf.fit(x_train, y_train)


def adaboost(x_train, y_train):
    clf = AdaBoostClassifier()
    return clf.fit(x_train, y_train)
