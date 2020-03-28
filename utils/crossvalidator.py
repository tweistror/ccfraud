import numpy as np

from sklearn.model_selection import StratifiedKFold

from baseline_methods.evaluate_sv_baselines import build_supervised_baselines


class Crossvalidator:
    def __init__(self, k, x, y):
        self.k = k
        self.x = x
        self.y = y

    def execute_cv(self):
        temp_prec_list = list()
        temp_reca_list = list()
        temp_f1_list = list()
        temp_acc_list = list()

        prec_cv_list = list()
        reca_cv_list = list()
        f1_cv_list = list()
        acc_cv_list = list()

        skf = StratifiedKFold(n_splits=self.k)

        for train_index, test_index in skf.split(self.x, self.y):
            x_train, x_test = self.x[train_index], self.x[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            precs, recas, f1s, accs, methods = \
                build_supervised_baselines(x_train, y_train, x_test, y_test)
            temp_prec_list.append(precs)
            temp_reca_list.append(recas)
            temp_f1_list.append(f1s)
            temp_acc_list.append(accs)

        temp_prec_list, temp_reca_list, temp_f1_list, temp_acc_list = \
            np.array(temp_prec_list), np.array(temp_reca_list), np.array(temp_f1_list), np.array(temp_acc_list)

        for index, method in enumerate(methods):
            prec = np.mean(temp_prec_list[:, index]).round(3)
            reca = np.mean(temp_reca_list[:, index]).round(3)
            f1 = np.mean(temp_f1_list[:, index]).round(3)
            acc = np.mean(temp_acc_list[:, index]).round(3)

            prec_cv_list.append(prec)
            reca_cv_list.append(reca)
            f1_cv_list.append(f1)
            acc_cv_list.append(acc)

        return prec_cv_list, reca_cv_list, f1_cv_list, acc_cv_list, methods
