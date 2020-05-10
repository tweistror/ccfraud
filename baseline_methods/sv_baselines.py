from sklearn import tree, ensemble, neighbors, svm, metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import accuracy_score, precision_recall_curve, precision_recall_fscore_support, \
    average_precision_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier

import xgboost as xgb
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


def binarize_test_labels(y_test):
    return [0 if val == 0 else 1 for val in y_test]


def get_metrics(y_test, y_pred, y_score, label, result_list):
    acc = accuracy_score(y_test, y_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, zero_division=0)

    precision_pts, recall_pts, _ = precision_recall_curve(y_test, y_score)

    # TODO: Small difference between metrics.auc(recall, precision) and average_precision_score(y_test, y_score)
    pr_auc = metrics.auc(recall_pts, precision_pts)
    # pr_auc = average_precision_score(y_test, y_score)

    roc_auc = roc_auc_score(y_test, y_score)

    result_list['prec_list'].append(precision[1])
    result_list['reca_list'].append(recall[1])
    result_list['f1_list'].append(fscore[1])
    result_list['acc_list'].append(acc)
    result_list['pr_auc_list'].append(pr_auc)
    result_list['roc_auc_list'].append(roc_auc)
    result_list['method_list'].append(label)

    return result_list


def plot_pr_curve(y_test, y_score, label):
    precision, recall, _ = precision_recall_curve(y_test, y_score)

    plt.plot(recall, precision, lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{label} - Precision-Recall-Curve - AUC: {metrics.auc(recall, precision)}')
    plt.show()


def plot_roc_curve(y_test, y_score, label):
    fpr, tpr, _ = metrics.roc_curve(y_test, y_score)

    plt.plot(fpr, tpr, lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{label} - Receiver Operating Characteristic - AUC: {metrics.auc(fpr, tpr)}')
    plt.show()


def execute_predict_proba(clf, train_test_split, label, result_list):
    x_train = train_test_split['x_train']
    y_train = train_test_split['y_train']
    x_test = train_test_split['x_test']
    y_test = train_test_split['y_test']

    clf.fit(x_train, y_train)

    y_test = binarize_test_labels(y_test)

    y_pred = clf.predict(x_test)
    y_score = clf.predict_proba(x_test)[:, 1]

    # plot_pr_curve(y_test, y_score, label)
    # plot_roc_curve(y_test, y_score, label)

    return get_metrics(y_test, y_pred, y_score, label, result_list)


def execute_decision_function(clf, train_test_split, label, result_list):
    x_train = train_test_split['x_train']
    y_train = train_test_split['y_train']
    x_test = train_test_split['x_test']
    y_test = train_test_split['y_test']

    clf.fit(x_train, y_train)

    y_test = binarize_test_labels(y_test)

    y_pred = clf.predict(x_test)
    y_score = clf.decision_function(x_test)

    # plot_pr_curve(y_test, y_score, label)
    # plot_roc_curve(y_test, y_score, label)

    return get_metrics(y_test, y_pred, y_score, label, result_list)


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
