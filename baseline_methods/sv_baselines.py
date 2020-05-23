from sklearn import tree, ensemble, neighbors, svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier

import xgboost as xgb
from sklearn.neural_network import MLPClassifier

from baseline_methods.utils import execute_decision_function, execute_predict_proba


class SV_Baselines:
    def __init__(self, image_creator, train_test_split):
        self.image_creator = image_creator
        self.train_test_split = train_test_split

    def svm_svc(self, label, result_list):
        clf = svm.SVC(kernel='rbf')

        return execute_decision_function(clf, self.train_test_split, label, result_list, self.image_creator)

    def knn(self, label, result_list):
        clf = neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance')

        return execute_predict_proba(clf, self.train_test_split, label, result_list, self.image_creator)

    def decision_tree(self, label, result_list):
        clf = tree.DecisionTreeClassifier()

        return execute_predict_proba(clf, self.train_test_split, label, result_list, self.image_creator)

    def random_forest(self, label, result_list):
        clf = ensemble.RandomForestClassifier()

        return execute_predict_proba(clf, self.train_test_split, label, result_list, self.image_creator)

    def svm_linearsvc(self, label, result_list):
        clf = svm.LinearSVC()

        return execute_decision_function(clf, self.train_test_split, label, result_list, self.image_creator)

    def gnb(self, label, result_list):
        clf = GaussianNB()

        return execute_predict_proba(clf, self.train_test_split, label, result_list, self.image_creator)

    def logistic_regression(self, label, result_list):
        clf = LogisticRegression(max_iter=1000)

        return execute_decision_function(clf, self.train_test_split, label, result_list, self.image_creator)

    def xgboost(self, label, result_list):
        clf = xgb.XGBClassifier(max_depth=10)

        return execute_predict_proba(clf, self.train_test_split, label, result_list, self.image_creator)

    def sgd(self, label, result_list):
        clf = SGDClassifier()

        return execute_decision_function(clf, self.train_test_split, label, result_list, self.image_creator)

    def gaussian_process(self, label, result_list):
        clf = GaussianProcessClassifier()

        return execute_predict_proba(clf, self.train_test_split, label, result_list, self.image_creator)

    def adaboost(self, label, result_list):
        clf = AdaBoostClassifier()

        return execute_decision_function(clf, self.train_test_split, label, result_list, self.image_creator)

    def mlp(self, label, result_list):
        clf = MLPClassifier()

        return execute_predict_proba(clf, self.train_test_split, label, result_list, self.image_creator)
