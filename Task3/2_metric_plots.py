#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

sns.set(
    context="notebook",
    style="whitegrid",
    rc={"figure.dpi": 120, "scatter.edgecolors": "k"},
)


def plot_pr_curve(y: np.ndarray, y_scores: np.ndarray):
    """Plots the precision recall curve given the true class labels y and the predicted labels with their probability"""
    '''https: // scikit - learn.org / stable / modules / generated / sklearn.metrics.precision_recall_curve.html'''
    precision, recall, thresholds = precision_recall_curve(y_true=y, probas_pred=y_scores[:, 1])

    # enumerate
    plt.figure()
    plt.plot(precision, recall, 'o', label = "Thresholds")
    for idx, threshold in enumerate(thresholds):
        plt.annotate(threshold, (precision[idx], recall[idx]))
    plt.xlabel('precision')
    plt.ylabel('recall')
    plt.title('Precision-Recall-Curve')
    plt.show()




def plot_roc_curve(y, y_scores):
    """Plots the Receiver Operating Characteristic Curve given the true class labels y and the predicted labels with
     their probability"""
    '''https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html'''
    '''Compute Receiver operating characteristic (ROC)'''
    fpr, tpr, thresholds = roc_curve(y_true=y, y_score=y_scores[:, 1])

    '''https: // scikit - learn.org / stable / modules / generated / sklearn.metrics.roc_auc_score.html'''
    '''Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction score'''
    auc = roc_auc_score(y, y_scores[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, 'o')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUC={auc:.2f})')
    plt.show()



def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray):
    """Plots the confusion matrix given the true test labels y_test and the predicted labes y_pred"""
    'https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html'
    'Compute confusion matrix to evaluate the accuracy of a classification.'
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, cmap=plt.cm.Greens)
    plt.title("Confusion matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.show()




def main():
    # Load Data
    X, y = datasets.load_breast_cancer(return_X_y=True)
    clf = RandomForestClassifier()

    # Compute cross validation probabilities for each sample
    # cross_val_predict:
    'https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html'
    '''Generate cross-validated estimates for each input data point. The data is split according to the cv parameter. Each sample belongs to exactly one test set, and its prediction is computed with an estimator fitted on the corresponding training set.'''
    'predictions = cross_val_predict(estimator, data to fit, target, ,verbositiy level=0'

    # cross_val_predict: returns predicted y values for the test fold.
    y_scores = cross_val_predict(
        estimator=clf, X=X, y=y, cv=5, n_jobs=-1, verbose=0, method="predict_proba"
    )

    plot_pr_curve(y, y_scores)
    plot_roc_curve(y, y_scores)

    # Load digits dataset
    X, y = datasets.load_digits(return_X_y=True)

    # Train/Test split
    'https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html'
    # train_test_split(array/list, testsize, trainingssize,

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )

    # Train classifier
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    plot_confusion_matrix(y_test, y_pred)


if __name__ == '__main__':
    main()
