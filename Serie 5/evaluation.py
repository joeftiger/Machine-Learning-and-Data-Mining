import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold


def kfold_eval(model, X: pd.DataFrame, y: pd.DataFrame, k: int, averaging=None) -> (
        np.array, np.array, np.array, np.array):
    """
    Performs k-fold cross-validation on the given model.

    :param model: The model to evaluate
    :param X: the features to train on
    :param y: the labels to predict
    :param k: how many folds to perform

    :param averaging: default='binary' This parameter is required for multiclass/multilabel targets. If ``None``,
    the scores for each class are returned. Otherwise, this determines the type of averaging performed on the data:
    ``'binary'``: Only report results for the class specified by ``pos_label``. This is applicable only if targets (
    ``y_{true,pred}``) are binary. ``'micro'``: Calculate metrics globally by counting the total true positives,
    false negatives and false positives. ``'macro'``: Calculate metrics for each label, and find their unweighted
    mean. This does not take label imbalance into account. ``'weighted'``: Calculate metrics for each label,
    and find their average weighted by support (the number of true instances for each label). This alters 'macro' to
    account for label imbalance; it can result in an F-score that is not between precision and recall. ``'samples'``:
    Calculate metrics for each instance, and find their average (only meaningful for multilabel classification where
    this differs from :func:`accuracy_score`).

    :return: accuracy, precision, recall, f1
    """
    if averaging is None:
        averaging = 'binary'

    kf = KFold(n_splits=k)

    accuracy = np.empty(kf.n_splits)
    precision = np.empty(kf.n_splits)
    recall = np.empty(kf.n_splits)
    f1 = np.empty(kf.n_splits)

    i = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        accuracy[i] = accuracy_score(pred, y_test)
        precision[i] = precision_score(pred, y_test, average=averaging)
        recall[i] = recall_score(pred, y_test, average=averaging)
        f1[i] = f1_score(pred, y_test, average=averaging)
        i += 1

    return accuracy, precision, recall, f1
