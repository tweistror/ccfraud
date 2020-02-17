from tabulate import tabulate
import numpy as np


def print_results(method_list, dataset_string, iteration_count, special_count, usv_count, prec_coll, reca_coll, f1_coll, acc_coll):
    results = list()

    print(f'{dataset_string}:  Average metrics over {iteration_count} iterations')

    for index, method in enumerate(method_list):
        if index == special_count - 1:
            results.append(['Special Methods'])
        if index == special_count:
            results.append(['Unsupervised Learning Methods'])
        if index == special_count + usv_count:
            results.append(['Supervised Learning Methods'])

        prec = f'{np.mean(prec_coll[:, index]).round(3)} \u00B1 {np.std(prec_coll[:, index]).round(3)}'
        reca = f'{np.mean(reca_coll[:, index]).round(3)} \u00B1 {np.std(reca_coll[:, index]).round(3)}'
        f1 = f'{np.mean(f1_coll[:, index]).round(3)} \u00B1 {np.std(f1_coll[:, index]).round(3)}'
        acc = f'{np.mean(acc_coll[:, index]).round(3)} \u00B1 {np.std(acc_coll[:, index]).round(3)}'
        results.append([method, prec, reca, f1, acc])

    print(tabulate(results, headers=['Method', 'Precision', 'Recall', 'F1 score', 'Accuracy']))
