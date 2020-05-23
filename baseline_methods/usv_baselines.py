from sklearn import covariance, svm, ensemble
from sklearn.neighbors import LocalOutlierFactor

from baseline_methods.utils import execute_decision_function


class USV_Baselines:
    def __init__(self, image_creator, train_test_split):
        self.image_creator = image_creator
        self.train_test_split = train_test_split

    def svm_oneclass(self, label, result_list):
        clf = svm.OneClassSVM()

        return execute_decision_function(clf, self.train_test_split, label, result_list, self.image_creator,
                                         unsupervised=True)

    def elliptic_envelope(self, label, result_list):
        clf = covariance.EllipticEnvelope(support_fraction=1)

        return execute_decision_function(clf, self.train_test_split, label, result_list, self.image_creator,
                                         unsupervised=True)

    def iso_forest(self, label, result_list):
        x_train = self.train_test_split['x_train']
        clf = ensemble.IsolationForest(max_samples=x_train.shape[0], random_state=None)

        return execute_decision_function(clf, self.train_test_split, label, result_list, self.image_creator,
                                         unsupervised=True)

    def local_outlier_factor(self, label, result_list):
        clf = LocalOutlierFactor(novelty=True)

        return execute_decision_function(clf, self.train_test_split, label, result_list, self.image_creator,
                                         unsupervised=True)
