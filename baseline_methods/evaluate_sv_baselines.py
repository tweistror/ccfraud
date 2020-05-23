from baseline_methods.sv_baselines import SV_Baselines
from baseline_methods.utils import binarize_sv_test_labels


def build_supervised_baselines(x_train, y_train, x_test, y_test, image_creator):
    results = {
        'prec_list': list(),
        'reca_list': list(),
        'f1_list': list(),
        'acc_list': list(),
        'pr_auc_list': list(),
        'roc_auc_list': list(),
        'method_list': list(),
    }

    train_test_split = {
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_test': binarize_sv_test_labels(y_test),
    }

    sv_baselines_class = SV_Baselines(image_creator, train_test_split)

    # SVM RBF SVC
    results = sv_baselines_class.svm_svc('SVM RBF SVC', results)

    # kNN
    results = sv_baselines_class.knn('kNN', results)

    # Decision Tree
    results = sv_baselines_class.decision_tree('Decision Tree', results)

    # Random Forest
    results = sv_baselines_class.random_forest('Random Forest', results)

    # SVM Linear SVC
    results = sv_baselines_class.svm_linearsvc('SVM Linear SVC', results)

    # Gaussian NB
    results = sv_baselines_class.gnb('Gaussian NB', results)

    # Logistic Regression
    results = sv_baselines_class.logistic_regression('Logistic Regression', results)

    # XGBoost
    results = sv_baselines_class.xgboost('XG Boost', results)

    # SGD Classifier
    results = sv_baselines_class.sgd('SGD', results)

    # Gaussian Process
    results = sv_baselines_class.gaussian_process('Gaussian Process', results)

    # Adaboost
    results = sv_baselines_class.adaboost('Adaboost', results)

    # Multi-Layer Perceptron
    results = sv_baselines_class.mlp('MLP', results)

    return results
