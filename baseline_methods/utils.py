import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_recall_curve, roc_auc_score


def binarize_sv_test_labels(y_test):
    return [0 if val == 0 else 1 for val in y_test]


def binarize_usv_test_labels(y_test):
    return [1 if val == 0 else -1 for val in y_test]


def get_metrics(y_test, y_pred, y_score, method, result_list):
    acc = accuracy_score(y_test, y_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, zero_division=0)

    precision_pts, recall_pts, _ = precision_recall_curve(y_test, y_score)

    pr_auc = metrics.auc(recall_pts, precision_pts)
    roc_auc = roc_auc_score(y_test, y_score)

    result_list['prec_list'].append(precision[1])
    result_list['reca_list'].append(recall[1])
    result_list['f1_list'].append(fscore[1])
    result_list['acc_list'].append(acc)
    result_list['pr_auc_list'].append(pr_auc)
    result_list['roc_auc_list'].append(roc_auc)
    result_list['method_list'].append(method)

    return result_list


def execute_predict_proba(clf, train_test_split, method, result_list, image_creator, unsupervised=False):
    x_train = train_test_split['x_train']
    y_train = train_test_split['y_train']
    x_test = train_test_split['x_test']
    y_test = train_test_split['y_test']

    if unsupervised is True:
        clf.fit(x_train)
    else:
        clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    y_score = clf.predict_proba(x_test)[:, 1]

    image_creator.add_baseline_curves(y_test, y_score, method, unsupervised)

    return get_metrics(y_test, y_pred, y_score, method, result_list)


def execute_decision_function(clf, train_test_split, method, result_list, image_creator, unsupervised=False):
    x_train = train_test_split['x_train']
    y_train = train_test_split['y_train']
    x_test = train_test_split['x_test']
    y_test = train_test_split['y_test']

    if unsupervised is True:
        clf.fit(x_train)
    else:
        clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    y_score = clf.decision_function(x_test)

    image_creator.add_baseline_curves(y_test, y_score, method, unsupervised)

    return get_metrics(y_test, y_pred, y_score, method, result_list)

