from baselines.classic_baselines import svm_svc, knn, random_forest, decision_tree, svm_linearsvc, gnb, xgboost, logistic_regression
from utils.run_models import run_classification


def build_classic_baselines(x_train, y_train, x_test, y_test):
    results = {
        'prec_list': list(),
        'reca_list': list(),
        'f1_list': list(),
        'acc_list': list(),
    }

    # SVM SVC
    results = evaluate_model(svm_svc(x_train, y_train), x_test, y_test, results)

    # kNN
    results = evaluate_model(knn(x_train, y_train), x_test, y_test, results)

    # Decision Tree
    results = evaluate_model(decision_tree(x_train, y_train), x_test, y_test, results)

    # Random Forest
    results = evaluate_model(random_forest(x_train, y_train), x_test, y_test, results)

    # SVM Linear SVC
    results = evaluate_model(svm_linearsvc(x_train, y_train), x_test, y_test, results)

    # Gaussian NB
    results = evaluate_model(gnb(x_train, y_train), x_test, y_test, results)

    # Logistic Regression
    results = evaluate_model(logistic_regression(x_train, y_train), x_test, y_test, results)

    # XGBoost
    results = evaluate_model(xgboost(x_train, y_train), x_test, y_test, results)

    return results['prec_list'], results['reca_list'], results['f1_list'], results['acc_list']


def evaluate_model(clf, x_test, y_test, lists):
    prec, reca, f1, acc = run_classification(x_test, y_test, clf, 'fraud-prediction')
    lists['prec_list'].append(prec)
    lists['reca_list'].append(reca)
    lists['f1_list'].append(f1)
    lists['acc_list'].append(acc)

    return lists
