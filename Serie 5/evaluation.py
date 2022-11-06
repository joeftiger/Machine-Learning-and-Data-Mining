from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold


def kfold_eval(model: Any, X: pd.DataFrame, y: pd.DataFrame, k: int, average: Any = 'binary',
               zero_division: Any = 'warn') -> (np.array, np.array, np.array, np.array):
    """
    Performs k-fold cross-validation on the given model.

    :param model: The model to evaluate
    :param X: the features to train on
    :param y: the labels to predict
    :param k: how many folds to perform
    :param average: default='binary' This parameter is required for multiclass/multilabel targets.
    :param zero_division: default='warn'

    :return: accuracy, precision, recall, f1
    """

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
        precision[i] = precision_score(pred, y_test, average=average, zero_division=zero_division)
        recall[i] = recall_score(pred, y_test, average=average, zero_division=zero_division)
        f1[i] = f1_score(pred, y_test, average=average, zero_division=zero_division)
        i += 1

    return accuracy, precision, recall, f1
