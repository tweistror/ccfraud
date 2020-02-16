from sklearn import tree, ensemble, neighbors, svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

import xgboost as xgb

def svm_svc(x_train, y_train):
    clf = svm.SVC()
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
