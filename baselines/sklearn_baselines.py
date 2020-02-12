from sklearn import svm


from baselines.sklearn_utils import get_sklearn_model_results


# SVM
def get_sklearn_svm_results(X_train, X_test, y_train, y_test):
    def train_svm(training_data, training_label):
        clf_svm = svm.SVC()
        clf_svm.fit(training_data, training_label)
        return clf_svm

    get_sklearn_model_results('Sklearn-SVM', train_svm, X_train, X_test, y_train, y_test)


