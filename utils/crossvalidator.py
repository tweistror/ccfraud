import numpy as np

from sklearn.model_selection import KFold, StratifiedKFold

from baseline_methods.evaluate_sv_baselines import build_supervised_baselines


class Crossvalidator:
    def __init__(self, k, cv_type, x, y):
        self.k = k
        self.cv_type = cv_type
        self.x = x
        self.y = y

    def execute_cv(self):
        temp_prec_list = list()
        temp_reca_list = list()
        temp_f1_list = list()
        temp_auc_list = list()

        prec_cv_list = list()
        reca_cv_list = list()
        f1_cv_list = list()
        auc_cv_list = list()

        if self.cv_type == 'StratifiedKFold':
            skf = StratifiedKFold(n_splits=self.k)

            for train_index, test_index in skf.split(self.x, self.y):
                x_train, x_test = self.x[train_index], self.x[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]
                precs, recas, f1s, aucs, methods = \
                    build_supervised_baselines(x_train, y_train, x_test, y_test)
                temp_prec_list.append(precs)
                temp_reca_list.append(recas)
                temp_f1_list.append(f1s)
                temp_auc_list.append(aucs)

        else:
            kf = KFold(n_splits=self.k)
            for train_index, test_index in kf.split(self.x):
                x_train, x_test = self.x[train_index], self.x[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]
                precs, recas, f1s, aucs, methods = \
                    build_supervised_baselines(x_train, y_train, x_test, y_test)
                temp_prec_list.append(precs)
                temp_reca_list.append(recas)
                temp_f1_list.append(f1s)
                temp_auc_list.append(aucs)

        temp_prec_list, temp_reca_list, temp_f1_list, temp_auc_list = \
            np.array(temp_prec_list), np.array(temp_reca_list), np.array(temp_f1_list), np.array(temp_auc_list)

        for index, method in enumerate(methods):
            prec = np.mean(temp_prec_list[:, index]).round(3)
            reca = np.mean(temp_reca_list[:, index]).round(3)
            f1 = np.mean(temp_f1_list[:, index]).round(3)
            auc = np.mean(temp_auc_list[:, index]).round(3)

            prec_cv_list.append(prec)
            reca_cv_list.append(reca)
            f1_cv_list.append(f1)
            auc_cv_list.append(auc)

        return prec_cv_list, reca_cv_list, f1_cv_list, auc_cv_list, methods
