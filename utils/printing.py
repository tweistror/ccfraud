from tabulate import tabulate
import numpy as np

from utils.to_latex_table import to_latex_table


def print_results(method_list, dataset_string, iteration_count, special_count, usv_count, prec_coll, reca_coll,
                  f1_coll, acc_coll, pr_auc_coll, roc_auc_coll, usv_train, sv_train, sv_train_fraud):
    results = list()
    l_results = list()

    for index, method in enumerate(method_list):
        if index == 0:
            results.append(['Special Methods'])
            l_results.append(['Special Methods'])
        if index == special_count:
            results.append(['Unsupervised Learning Methods'])
            l_results.append(['Unsupervised Learning Methods'])
        if index == special_count + usv_count:
            results.append(['Supervised Learning Methods'])
            l_results.append(['Supervised Learning Methods'])

        prec = f'{np.mean(prec_coll[:, index]).round(3)} \u00B1 {np.std(prec_coll[:, index]).round(3)}'
        reca = f'{np.mean(reca_coll[:, index]).round(3)} \u00B1 {np.std(reca_coll[:, index]).round(3)}'
        f1 = f'{np.mean(f1_coll[:, index]).round(3)} \u00B1 {np.std(f1_coll[:, index]).round(3)}'
        acc = f'{np.mean(acc_coll[:, index]).round(3)} \u00B1 {np.std(acc_coll[:, index]).round(3)}'
        pr_auc = f'{np.mean(pr_auc_coll[:, index]).round(3)} \u00B1 {np.std(pr_auc_coll[:, index]).round(3)}'
        roc_auc = f'{np.mean(roc_auc_coll[:, index]).round(3)} \u00B1 {np.std(roc_auc_coll[:, index]).round(3)}'

        # TODO: Add pr_auc and roc_auc
        # For latex table
        l_prec = f'{np.mean(prec_coll[:, index]).round(3)} $\\pm$ {np.std(prec_coll[:, index]).round(3)}'
        l_reca = f'{np.mean(reca_coll[:, index]).round(3)} $\\pm$ {np.std(reca_coll[:, index]).round(3)}'
        l_f1 = f'{np.mean(f1_coll[:, index]).round(3)} $\\pm$ {np.std(f1_coll[:, index]).round(3)}'
        l_acc = f'{np.mean(acc_coll[:, index]).round(3)} $\\pm$ {np.std(acc_coll[:, index]).round(3)}'

        results.append([method, prec, reca, f1, acc, pr_auc, roc_auc])
        l_results.append([method, l_prec, l_reca, l_f1, l_acc])

    # Script for creating latex tables
    # to_latex_table(dataset_string, l_results, ['Method', 'Precision', 'Recall', 'F1 score', 'ACC'],
    #                usv_train, sv_train, sv_train_fraud)

    print(f'{dataset_string}:  Average metrics over {iteration_count} iterations')

    print('Training Information:')
    print(f'usv_train: {usv_train} | sv_train: {sv_train - sv_train_fraud} benign & {sv_train_fraud} fraud')
    print(tabulate(results, headers=['Method', 'Precision', 'Recall', 'F1 score', 'ACC', 'PR AUC', 'ROC AUC']))
