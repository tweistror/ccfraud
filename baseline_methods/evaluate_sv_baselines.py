from baseline_methods.sv_baselines import svm_svc, knn, random_forest, decision_tree, svm_linearsvc, gnb, xgboost, \
    logistic_regression, sgd, gaussian_process, adaboost, mlp


def build_supervised_baselines(x_train, y_train, x_test, y_test):
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
        'y_test': y_test,
    }

    # SVM RBF SVC
    results = svm_svc(train_test_split, 'SVM RBF SVC', results)

    # kNN
    results = knn(train_test_split, 'kNN', results)

    # Decision Tree
    results = decision_tree(train_test_split, 'Decision Tree', results)

    # Random Forest
    results = random_forest(train_test_split, 'Random Forest', results)

    # SVM Linear SVC
    results = svm_linearsvc(train_test_split, 'SVM Linear SVC', results)

    # Gaussian NB
    results = gnb(train_test_split, 'Gaussian NB', results)

    # Logistic Regression
    results = logistic_regression(train_test_split, 'Logistic Regression', results)

    # XGBoost
    results = xgboost(train_test_split, 'XG Boost', results)

    # SGD Classifier
    results = sgd(train_test_split, 'SGD', results)

    # Gaussian Process
    results = gaussian_process(train_test_split, 'Gaussian Process', results)

    # Adaboost
    results = adaboost(train_test_split, 'Adaboost', results)

    # Multi-Layer Perceptron
    results = mlp(train_test_split, 'MLP', results)

    return results
