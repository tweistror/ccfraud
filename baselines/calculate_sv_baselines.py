from xgboost import plot_importance
import numpy as np

from baselines.sv_baselines import svm_svc, knn, random_forest, decision_tree, svm_linearsvc, gnb, xgboost, \
    logistic_regression, sgd, gaussian_process, adaboost, mlp
from utils.run_models import run_sv_classification
import matplotlib.pyplot as plt


def build_supervised_baselines(x_train, y_train, x_test, y_test):

    def evaluate_model(clf, lists, label):
        prec, reca, f1, auc = run_sv_classification(x_test, y_test, clf, 'fraud-prediction')
        lists['prec_list'].append(prec)
        lists['reca_list'].append(reca)
        lists['f1_list'].append(f1)
        lists['auc_list'].append(auc)
        lists['method_list'].append(label)

        return lists

    results = {
        'prec_list': list(),
        'reca_list': list(),
        'f1_list': list(),
        'auc_list': list(),
        'method_list': list()
    }

    # SVM RBF SVC
    results = evaluate_model(svm_svc(x_train, y_train), results, 'SVM RBF SVC')

    # kNN
    results = evaluate_model(knn(x_train, y_train), results, 'kNN')

    # Decision Tree
    results = evaluate_model(decision_tree(x_train, y_train), results, 'Decision Tree')

    # Random Forest
    results = evaluate_model(random_forest(x_train, y_train), results, 'Random Forest')

    # SVM Linear SVC
    results = evaluate_model(svm_linearsvc(x_train, y_train), results, 'SVM Linear SVC')

    # Gaussian NB
    results = evaluate_model(gnb(x_train, y_train), results, 'Gaussian NB')

    # Logistic Regression
    results = evaluate_model(logistic_regression(x_train, y_train), results, 'Logistic Regression')

    # XGBoost
    clf = xgboost(x_train, y_train)
    results = evaluate_model(clf, results, 'XG Boost')
    # plot_xgb_feature_importance(clf)

    # SGD Classifier
    results = evaluate_model(sgd(x_train, y_train), results, 'SGD')

    # Gaussian Process
    results = evaluate_model(gaussian_process(x_train, y_train), results, 'Gaussian Process')

    # Adaboost
    results = evaluate_model(adaboost(x_train, y_train), results, 'Adaboost')

    # Multi-Layer Perceptron
    results = evaluate_model(mlp(x_train, y_train), results, 'MLP')

    return results['prec_list'], results['reca_list'], results['f1_list'], results['auc_list'], results['method_list']


def plot_xgb_feature_importance(clf):
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111)

    colours = plt.cm.Set1(np.linspace(0, 1, 9))

    ax = plot_importance(clf, height=1, color=colours, grid=False, \
                         show_values=False, importance_type='cover', ax=ax);
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)

    ax.set_xlabel('importance score', size=16);
    ax.set_ylabel('features', size=16);
    ax.set_yticklabels(ax.get_yticklabels(), size=12);
    ax.set_title('Ordering of features by importance to the model learnt', size=20)

    plt.show()
