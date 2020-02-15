import numpy as np

from sklearn.metrics import classification_report, precision_recall_fscore_support


def run_one_svm(x_test, y_test, clf, mode):
    n = 490
    x_test_svm = np.concatenate((x_test[y_test == 0][0:n], x_test[y_test == 1][0:n]))
    y_test_svm = np.concatenate((np.ones(n), np.zeros(n)-1))
    y_pred = clf.predict(x_test_svm)
    # class_report = classification_report(y_test_svm, y_pred, target_names=['fraud', 'benign'], digits=4)
    acc = np.sum(y_pred == y_test_svm) / float(y_pred.shape[0])

    precision, recall, fscore, support = precision_recall_fscore_support(y_test_svm, y_pred)

    return precision[0], recall[0], fscore[0], acc
