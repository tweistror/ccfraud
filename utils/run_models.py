import numpy as np

from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix


def run_usv_classification(x_test, y_test, clf, negative_samples, mode):
    # TODO: Conditional statements for different modes (fraud and benign)
    x_test_svm = np.concatenate((x_test[y_test == 0][0:negative_samples], x_test[y_test == 1][0:negative_samples]))
    y_test_svm = np.concatenate((np.ones(negative_samples), np.zeros(negative_samples)-1))
    y_pred = clf.predict(x_test_svm)
    # class_report = classification_report(y_test_svm, y_pred, target_names=['fraud', 'benign'], digits=4)
    acc = np.sum(y_pred == y_test_svm) / float(y_pred.shape[0])

    precision, recall, fscore, support = precision_recall_fscore_support(y_test_svm, y_pred, zero_division=0)

    return precision[0], recall[0], fscore[0], acc


def run_classification(x_test, y_test, clf, negative_samples, mode):
    # TODO: Conditional statements for different modes (fraud and benign)
    x_test = np.concatenate((x_test[y_test == 0][0:negative_samples], x_test[y_test == 1][0:negative_samples]))
    y_test = np.concatenate((np.zeros(negative_samples), np.ones(negative_samples)))

    y_pred = clf.predict(x_test)
    # class_report = classification_report(y_test_svm, y_pred, target_names=['fraud', 'benign'], digits=4)
    # cm = confusion_matrix(y_test, y_pred)
    acc = np.sum(y_pred == y_test) / float(y_pred.shape[0])

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, zero_division=0)

    return precision[0], recall[0], fscore[0], acc