from baseline_methods.sv_baselines import svm_svc, knn, random_forest, decision_tree, svm_linearsvc, gnb, xgboost, \
    logistic_regression, sgd, gaussian_process, adaboost, mlp
from baseline_methods.run_models import run_sv_classification


def build_supervised_baselines(x_train, y_train, x_test, y_test):

    def evaluate_model(clf, lists, label):
        prec, reca, f1, acc = run_sv_classification(x_test, y_test, clf, 'fraud-prediction')
        lists['prec_list'].append(prec)
        lists['reca_list'].append(reca)
        lists['f1_list'].append(f1)
        lists['acc_list'].append(acc)
        lists['method_list'].append(label)

        return lists

    results = {
        'prec_list': list(),
        'reca_list': list(),
        'f1_list': list(),
        'acc_list': list(),
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
    results = evaluate_model(xgboost(x_train, y_train), results, 'XG Boost')

    # SGD Classifier
    results = evaluate_model(sgd(x_train, y_train), results, 'SGD')

    # Gaussian Process
    results = evaluate_model(gaussian_process(x_train, y_train), results, 'Gaussian Process')

    # Adaboost
    results = evaluate_model(adaboost(x_train, y_train), results, 'Adaboost')

    # Multi-Layer Perceptron
    results = evaluate_model(mlp(x_train, y_train), results, 'MLP')

    return results['prec_list'], results['reca_list'], results['f1_list'], results['acc_list'], results['method_list']
