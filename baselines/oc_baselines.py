from sklearn import covariance, svm, ensemble
from sklearn.neighbors import LocalOutlierFactor


def svm_oneclass(x):
    clf = svm.OneClassSVM()
    return clf.fit(x)


def elliptic_envelope(x):
    clf = covariance.EllipticEnvelope(support_fraction=1)
    return clf.fit(x)


def iso_forest(x):
    clf = ensemble.IsolationForest(max_samples=x.shape[0], random_state=None)
    return clf.fit(x)


def local_outlier_factor(x):
    clf = LocalOutlierFactor(novelty=True)
    return clf.fit(x)
