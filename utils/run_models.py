import numpy as np

from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix


def run_usv_classification(x_test, y_test, clf, negative_samples, mode):
    # TODO: Conditional statements for different modes (fraud and benign)
    x_test_svm = np.concatenate((x_test[y_test == 0][0:negative_samples], x_test[y_test == 1][0:negative_samples]))
    y_test_svm = np.concatenate((np.ones(negative_samples), np.zeros(negative_samples) - 1))
    y_pred = clf.predict(x_test_svm)
    acc = np.sum(y_pred == y_test_svm) / float(y_pred.shape[0])

    # cm = confusion_matrix(y_test_svm, y_pred)
    # class_report = classification_report(y_test_svm, y_pred, target_names=['benign', 'fraud'], digits=4)

    precision, recall, fscore, support = precision_recall_fscore_support(y_test_svm, y_pred, zero_division=0)

    return precision[1], recall[1], fscore[1], acc


def run_sv_classification(x_test, y_test, clf, negative_samples, mode):
    # TODO: Conditional statements for different modes (fraud and benign)
    x_test = np.concatenate((x_test[y_test == 0][0:negative_samples], x_test[y_test == 1][0:negative_samples]))
    y_test = np.concatenate((np.zeros(negative_samples), np.ones(negative_samples)))

    y_pred = clf.predict(x_test)
    acc = np.sum(y_pred == y_test) / float(y_pred.shape[0])

    # cm = confusion_matrix(y_test, y_pred)
    # class_report = classification_report(y_test, y_pred, target_names=['benign', 'fraud'], digits=4)

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, zero_division=0)

    return precision[1], recall[1], fscore[1], acc
