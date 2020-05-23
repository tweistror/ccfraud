import numpy as np

from sklearn.model_selection import StratifiedKFold

from baseline_methods.evaluate_sv_baselines import build_supervised_baselines


class Crossvalidator:
    def __init__(self, k, x, y, image_creator, seed):
        self.k = k
        self.x = x
        self.y = y
        self.image_creator = image_creator
        self.seed = seed

    def execute_cv(self, x_test, y_test):
        temp_prec_list = list()
        temp_reca_list = list()
        temp_f1_list = list()
        temp_acc_list = list()
        temp_pr_auc_list = list()
        temp_roc_auc_list = list()

        x, y = self.x, self.y

        skf = StratifiedKFold(n_splits=self.k)

        first_iteration = True

        for train_index, test_index in skf.split(x, y):
            x_train, _ = x[train_index], x[test_index]
            y_train, _ = y[train_index], y[test_index]

            results = build_supervised_baselines(x_train, y_train, x_test, y_test, self.image_creator)

            if first_iteration is True:
                method_list = results['method_list']
                first_iteration = False

            temp_prec_list.append(results['prec_list'])
            temp_reca_list.append(results['reca_list'])
            temp_f1_list.append(results['f1_list'])
            temp_acc_list.append(results['acc_list'])
            temp_pr_auc_list.append(results['pr_auc_list'])
            temp_roc_auc_list.append(results['roc_auc_list'])

        temp_prec_list, temp_reca_list, temp_f1_list, temp_acc_list, temp_pr_auc_list, temp_roc_auc_list = \
            np.array(temp_prec_list), np.array(temp_reca_list), np.array(temp_f1_list), np.array(temp_acc_list), \
            np.array(temp_pr_auc_list), np.array(temp_roc_auc_list)

        results = {
            'prec_list': list(),
            'reca_list': list(),
            'f1_list': list(),
            'acc_list': list(),
            'pr_auc_list': list(),
            'roc_auc_list': list(),
            'method_list': list(),
        }

        for index, method in enumerate(method_list):
            prec = np.mean(temp_prec_list[:, index]).round(3)
            reca = np.mean(temp_reca_list[:, index]).round(3)
            f1 = np.mean(temp_f1_list[:, index]).round(3)
            acc = np.mean(temp_acc_list[:, index]).round(3)
            pr_auc = np.mean(temp_pr_auc_list[:, index]).round(3)
            roc_auc = np.mean(temp_roc_auc_list[:, index]).round(3)

            results['prec_list'].append(prec)
            results['reca_list'].append(reca)
            results['f1_list'].append(f1)
            results['acc_list'].append(acc)
            results['pr_auc_list'].append(pr_auc)
            results['roc_auc_list'].append(roc_auc)
            results['method_list'] = method_list

        return results
