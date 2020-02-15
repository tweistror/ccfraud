from sklearn import tree, ensemble, neighbors, svm, covariance

from baselines.sklearn_utils import get_sklearn_model_results


# SVM
def svm_svc(X_train, y_train):
    clf = svm.SVC()
    return clf.fit(X_train, y_train)


def knn(X_train, y_train):
    clf = neighbors.KNeighborsClassifier(n_neighbors=3)
    return clf.fit(X_train, y_train)


def decision_tree(X_train, y_train):
    clf = tree.DecisionTreeClassifier()
    return clf.fit(X_train, y_train)


def random_forest(X_train, y_train):
    clf = ensemble.RandomForestClassifier()
    return clf.fit(X_train, y_train)


def svm_nusvc(X, y):
    clf = svm.NuSVC()
    return clf.fit(X, y)


def svm_linearsvc(X, y):
    clf = svm.LinearSVC()
    return clf.fit(X, y)


def svm_oneclass(X):
    clf = svm.OneClassSVM()
    return clf.fit(X)


def elliptic_envelope(X):
    clf = covariance.EllipticEnvelope()
    return clf.fit(X)


def iso_forest(X):
    clf = ensemble.IsolationForest(max_samples=X.shape[0], random_state=None)
    return clf.fit(X)
