# Utility imports
import datetime
import itertools

import numpy as np

# Plotting imports
import matplotlib.pyplot as plt
from matplotlib import rcParams, gridspec

# sklearn imports
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def get_sklearn_model_results(model_name, train_fct, X_train, X_test, y_train, y_test):
    # Set starting time
    start_time = datetime.now()

    # Train model
    trained_model = train_fct(X_train, y_train)
    # Predict train and test data
    pred_train = trained_model.predict(X_train)
    pred_test = trained_model.predict(X_test)
    # Calculate scores
    accuracy = accuracy_score(y_test, pred_test)
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, pred_test, average='binary')
    time_required = datetime.now() - start_time
    print('sklearn {} results:\nAccuracy: {}\nPrecision: {}\nRecall: {}\nF-Score: {}'.
          format(model_name, accuracy * 100, precision * 100, recall * 100, fscore * 100))
    print('Data information:')
    print('Time required: {}\n\n'.format(time_required))
    train_cm = confusion_matrix(y_train, pred_train)
    test_cm = confusion_matrix(y_test, pred_test)
    plot_confusion_normal(train_cm, test_cm, name=model_name)


def plot_confusion_normal(train_matrix, test_matrix, name, classes=[0, 1], cmap=plt.cm.Greens):
    # Set the plot size
    rcParams['figure.figsize'] = (30.0, 22.5)

    # Set up grid
    plt.figure()
    fig = gridspec.GridSpec(3, 3)
    grid_length = list(range(1, 3))
    tuple_grid = [(i, j) for i in grid_length for j in grid_length]

    # Plot Training Confusion Matrix
    plt.subplot2grid((3, 3), (0, 0))
    plot_confusion_matrix(
        cm=train_matrix,
        classes=classes,
        title=name.capitalize() + ': Train Set',
        cmap=cmap)

    # Plot Testing Confusion Matrix
    plt.subplot2grid((3, 3), (0, 1))
    plot_confusion_matrix(
        cm=test_matrix,
        classes=classes,
        title=name.capitalize() + ': Test Set',
        cmap=cmap)

    return None


def plot_confusion_matrix(cm, classes, fontsize=20,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    cm_num = cm
    cm_per = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title.replace('_', ' ').title() + '\n', size=fontsize)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=fontsize)
    plt.yticks(tick_marks, classes, size=fontsize)

    fmt = '.5f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # Set color parameters
        color = "white" if cm[i, j] > thresh else "black"
        alignment = "center"

        # Plot perentage
        text = format(cm_per[i, j], '.5f')
        text = text + '%'
        plt.text(j, i,
                 text,
                 fontsize=fontsize,
                 verticalalignment='baseline',
                 horizontalalignment='center',
                 color=color)
        # Plot numeric
        text = format(cm_num[i, j], 'd')
        text = '\n \n' + text
        plt.text(j, i,
                 text,
                 fontsize=fontsize,
                 verticalalignment='center',
                 horizontalalignment='center',
                 color=color)

    plt.tight_layout()
    plt.ylabel('True label'.title(), size=fontsize)
    plt.xlabel('Predicted label'.title(), size=fontsize)

    return None
