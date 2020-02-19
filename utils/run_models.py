import numpy as np

from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix


def run_usv_classification(x_test, y_test, clf, mode):
    n = int(len(y_test) / 2)

    # TODO: Conditional statements for different modes (fraud and benign)
    # Label for inliers/outliers have to be 1/-1 in usv
    y_test[:n] = 1  # Benign
    y_test[n:] = -1  # Fraud
    y_pred = clf.predict(x_test)
    acc = np.sum(y_pred == y_test) / float(y_pred.shape[0])

    # cm = confusion_matrix(y_test, y_pred)
    # class_report = classification_report(y_test, y_pred, target_names=['benign', 'fraud'], digits=4)

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, zero_division=0)

    return precision[1], recall[1], fscore[1], acc


def run_sv_classification(x_test, y_test, clf, mode):
    n = int(len(y_test) / 2)

    # TODO: Conditional statements for different modes (fraud and benign)

    y_test[:n] = 0  # Benign
    y_test[n:] = 1  # Fraud

    y_pred = clf.predict(x_test)
    acc = np.sum(y_pred == y_test) / float(y_pred.shape[0])

    # cm = confusion_matrix(y_test, y_pred)
    # class_report = classification_report(y_test, y_pred, target_names=['benign', 'fraud'], digits=4)

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, zero_division=0)

    return precision[1], recall[1], fscore[1], acc
