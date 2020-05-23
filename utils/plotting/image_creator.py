import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve
from xgboost import plot_importance
import random
from datetime import datetime
from statistics import mean


def get_dataset_name(dataset_string):
    if dataset_string == "paysim":
        return 'PaySim Kaggle'
    if dataset_string == "paysim-custom":
        return 'PaySim Custom'
    elif dataset_string == "ccfraud":
        return 'CC Fraud Kaggle'
    elif dataset_string == "ieee":
        return 'IEEE Kaggle'
    elif dataset_string == "nslkdd":
        return 'NSL-KDD'
    elif dataset_string == "saperp-ek":
        return 'Synthetic SAPERP-EK'
    elif dataset_string == "saperp-vk":
        return 'Synthetic SAPERP-VK'
    elif dataset_string == "mnist":
        return 'MNIST'
    elif dataset_string == "cifar10":
        return 'CIFAR10'
        
    return dataset_string


def merge_curves(x_label, y_label, curve_dict, curve_list):
    for method in curve_dict:
        x_values = []
        y_values = []
        auc_values = []
        for curve in curve_dict[method]:
            x_values.append(curve[x_label])
            y_values.append(curve[y_label])
            auc_values.append(metrics.auc(curve[x_label], curve[y_label]))

        average_x = [np.mean(k) for k in zip(*x_values)]
        average_y = [np.mean(k) for k in zip(*y_values)]
        average_auc = np.mean(auc_values)

        curve_list.append({x_label: average_x, y_label: average_y, 'auc': average_auc, 'method': method})

    return curve_list


class Image_Creator:
    def __init__(self, dataset_string, baseline):
        self.dataset_string = dataset_string
        self.baseline = baseline
        self.roc_curve_list = []
        self.pr_curve_list = []

        self.baseline_pr_curve_list = []
        self.baseline_roc_curve_list = []

    def create_plots(self):
        method_list = []
        baseline_sv_methods = []
        baseline_usv_methods = []

        for pr_curve in self.pr_curve_list:
            if pr_curve['method'] not in method_list:
                method_list.append(pr_curve['method'])

        for method in method_list:
            self.__plot_pr_curves(method)
            self.__plot_roc_curves(method)

        for curve in self.baseline_pr_curve_list:
            if curve['baseline_type'] == 'Supervised':
                if curve['method'] not in baseline_sv_methods:
                    baseline_sv_methods.append(curve['method'])
            if curve['baseline_type'] == 'Unsupervised':
                if curve['method'] not in baseline_usv_methods:
                    baseline_usv_methods.append(curve['method'])

        # Dicts with baseline methods as keys and curves as values
        sv_pr_dict = {}
        sv_roc_dict = {}
        usv_pr_dict = {}
        usv_roc_dict = {}

        for method in baseline_sv_methods:
            sv_pr_list = []
            sv_roc_list = []
            for curve in self.baseline_pr_curve_list:
                if curve['method'] == method:
                    sv_pr_list.append(curve)
            for curve in self.baseline_roc_curve_list:
                if curve['method'] == method:
                    sv_roc_list.append(curve)
            sv_pr_dict[method] = sv_pr_list
            sv_roc_dict[method] = sv_roc_list

        for method in baseline_usv_methods:
            usv_pr_list = []
            usv_roc_list = []
            for curve in self.baseline_pr_curve_list:
                if curve['method'] == method:
                    usv_pr_list.append(curve)
            for curve in self.baseline_roc_curve_list:
                if curve['method'] == method:
                    usv_roc_list.append(curve)
            usv_pr_dict[method] = usv_pr_list
            usv_roc_dict[method] = usv_roc_list

        # Lists holding averaged pr and roc curves
        pr_curves = []
        roc_curves = []

        if self.baseline == 'sv' or self.baseline == 'both':
            pr_curves = merge_curves('recall', 'precision', sv_pr_dict, pr_curves)
            roc_curves = merge_curves('fpr', 'tpr', sv_roc_dict, roc_curves)

        if self.baseline == 'usv' or self.baseline == 'both':
            pr_curves = merge_curves('recall', 'precision', usv_pr_dict, pr_curves)
            roc_curves = merge_curves('fpr', 'tpr', usv_roc_dict, roc_curves)

        self.__plot_baseline_pr_curves(pr_curves)
        self.__plot_baseline_roc_curves(roc_curves)

    def __plot_baseline_pr_curves(self, pr_curves):
        plt.clf()

        multiple_lists = [[2, 5, 1, 9], [4, 9, 5, 10]]
        arrays = [np.array(x) for x in multiple_lists]
        [np.mean(k) for k in zip(*arrays)]

        for pr_curve in pr_curves:
            plt.plot(pr_curve['recall'], pr_curve['precision'],
                     label="{}, AUC={:.3f}".format(pr_curve['method'], pr_curve['auc']))

        plt.plot([1, 0], [0, 1], color='orange', linestyle='--')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Baseline Methods - PR-Curves')

        plt.legend(prop={'size': 8}, loc='lower left')

        timestamp = datetime.now().isoformat(timespec='seconds').replace(':', '.')
        date, time = timestamp.split('T')

        results_dir = os.path.join('output/', f'{self.dataset_string}/', f'{date}/')

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        plt.savefig(f'output/{self.dataset_string}/{date}/pr_baselines_{time}.png', dpi=400)

    def __plot_baseline_roc_curves(self, roc_curves):
        plt.clf()

        for roc_curve in roc_curves:
            plt.plot(roc_curve['fpr'], roc_curve['tpr'],
                     label="{}, AUC={:.3f}".format(roc_curve['method'], roc_curve['auc']))

        plt.plot([0, 1], [0, 1], color='orange', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Baseline Methods - ROC-Curves')

        plt.legend(prop={'size': 8}, loc='lower right')

        timestamp = datetime.now().isoformat(timespec='seconds').replace(':', '.')
        date, time = timestamp.split('T')

        results_dir = os.path.join('output/', f'{self.dataset_string}/', f'{date}/')

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        plt.savefig(f'output/{self.dataset_string}/{date}/roc_baselines_{time}.png', dpi=400)

    def add_curves(self, y_test, y_score, method):
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        fpr, tpr, _ = metrics.roc_curve(y_test, y_score)

        self.pr_curve_list.append({'precision': precision, 'recall': recall, 'method': method})
        self.roc_curve_list.append({'fpr': fpr, 'tpr': tpr, 'method': method})

    def add_baseline_curves(self, y_test, y_score, method, unsupervised=False):
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        fpr, tpr, _ = metrics.roc_curve(y_test, y_score)

        self.baseline_pr_curve_list.append({'precision': precision, 'recall': recall, 'method': method,
                                            'baseline_type': 'Supervised' if unsupervised is False else 'Unsupervised'})
        self.baseline_roc_curve_list.append({'fpr': fpr, 'tpr': tpr, 'method': method,
                                            'baseline_type': 'Supervised' if unsupervised is False else 'Unsupervised'})

    def __plot_pr_curves(self, method):
        pr_auc_list = []
        plt.clf()

        for pr_curve in self.pr_curve_list:
            if method == pr_curve['method']:
                plt.plot(pr_curve['recall'], pr_curve['precision'])
                pr_auc_list.append(metrics.auc(pr_curve['recall'], pr_curve['precision']))

        plt.plot([1, 0], [0, 1], color='orange', linestyle='--')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{self.dataset_string} - {method} - PR-Curve - AUC: {mean(pr_auc_list).round(3)}')

        timestamp = datetime.now().isoformat(timespec='seconds').replace(':', '.')
        date, time = timestamp.split('T')

        results_dir = os.path.join('output/', f'{self.dataset_string}/', f'{date}/')

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        plt.savefig(f'output/{self.dataset_string}/{date}/pr_{method}_{time}.png')

    def __plot_roc_curves(self, method):
        roc_auc_list = []
        plt.clf()

        for roc_curve in self.roc_curve_list:
            if method == roc_curve['method']:
                plt.plot(roc_curve['fpr'], roc_curve['tpr'])
                roc_auc_list.append(metrics.auc(roc_curve['fpr'], roc_curve['tpr']))

        plt.plot([0, 1], [0, 1], color='orange', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{self.dataset_string} - {method} - ROC-Curve - AUC: {mean(roc_auc_list).round(3)}')

        timestamp = datetime.now().isoformat(timespec='seconds').replace(':', '.')
        date, time = timestamp.split('T')

        results_dir = os.path.join('output/', f'{self.dataset_string}/', f'{date}/')

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        plt.savefig(f'output/{self.dataset_string}/{date}/roc_{method}_{time}.png')

    def plot_mnist_images(self, x_test, x_generated, label, dataset='', n=5):

        indices = random.sample(range(0, x_test.shape[0] - 1), n)

        dim = 28

        plt.figure(figsize=(10, 4.5))

        for i in range(n):
            # plot original image
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_test[indices[i]].reshape(dim, dim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == n/2:
                ax.set_title('Original Images')

            # plot reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(x_generated[indices[i]].reshape(dim, dim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == n/2:
                ax.set_title('Reconstructed Images')
        plt.show()

    def plot_cifar10_images(self, x_test, x_generated, label, dataset='', n=5):
        indices = random.sample(range(0, x_test.shape[0] - 1), n)

        plt.figure(figsize=(10, 4.5))

        for i in range(n):
            # plot original image
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_test[indices[i]].reshape(3, 32, 32).transpose([1, 2, 0]))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == n/2:
                ax.set_title('Original Images')

            # plot reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(x_generated[indices[i]].reshape(3, 32, 32).transpose([1, 2, 0]))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == n/2:
                ax.set_title('Reconstructed Images')
        plt.show()

    def tsne_plot(self, x1, y1, name="graph.png"):
        tsne = TSNE(n_components=2, random_state=0)
        X_t = tsne.fit_transform(x1)
        #     plt.figure(figsize=(12, 8))
        plt.scatter(X_t[np.where(y1 == 0), 0], X_t[np.where(y1 == 0), 1], marker='o', color='g', linewidth='1', alpha=0.8,
                    label='Non Fraud', s=2)
        plt.scatter(X_t[np.where(y1 == 1), 0], X_t[np.where(y1 == 1), 1], marker='o', color='r', linewidth='1', alpha=0.8,
                    label='Fraud', s=2)

        plt.legend(loc='best')
        plt.savefig(name)
        plt.show()

    # Plot Keras training history
    def plot_loss(self, hist):
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.yscale('log', basey=10)
        plt.show()

    def plot_xgb_feature_importance(self, clf):
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
