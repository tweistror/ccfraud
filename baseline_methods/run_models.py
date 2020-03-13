from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix, roc_auc_score


def run_usv_classification(x_test, y_test, clf, mode):
    y_test = [1 if val == 0 else -1 for val in y_test]

    y_pred = clf.predict(x_test)
    auc_score = roc_auc_score(y_test, y_pred)

    # cm = confusion_matrix(y_test, y_pred)
    # class_report = classification_report(y_test, y_pred, target_names=['benign', 'fraud'], digits=4)

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, zero_division=0)

    return precision[1], recall[1], fscore[1], auc_score


def run_sv_classification(x_test, y_test, clf, mode):
    y_test = [0 if val == 0 else 1 for val in y_test]

    y_pred = clf.predict(x_test)
    auc_score = roc_auc_score(y_test, y_pred)

    # cm = confusion_matrix(y_test, y_pred)
    # class_report = classification_report(y_test, y_pred, target_names=['benign', 'fraud'], digits=4)

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, zero_division=0)

    return precision[1], recall[1], fscore[1], auc_score
