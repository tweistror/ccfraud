from sklearn import covariance, svm, ensemble


def svm_oneclass(x):
    clf = svm.OneClassSVM()
    return clf.fit(x)


def elliptic_envelope(x):
    clf = covariance.EllipticEnvelope()
    return clf.fit(x)


def iso_forest(x):
    clf = ensemble.IsolationForest(max_samples=x.shape[0], random_state=None)
    return clf.fit(x)
