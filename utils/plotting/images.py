import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from xgboost import plot_importance
import math
import random


def plot_mnist_images(x_test, x_generated, label, dataset='', n=5):

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


def plot_cifar10_images(x_test, x_generated, label, dataset='', n=5):
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


def tsne_plot(x1, y1, name="graph.png"):
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
def plot_loss(hist):
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.yscale('log', basey=10)
    plt.show()


def plot_xgb_feature_importance(clf):
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
