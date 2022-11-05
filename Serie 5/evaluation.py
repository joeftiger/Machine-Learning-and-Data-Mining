import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold


def kfold_eval(model, X: pd.DataFrame, y: pd.DataFrame, k: int) -> (np.array, np.array, np.array, np.array):
    """
    Performs k-fold cross-validation on the given model.

    :param model: The model to evaluate
    :param X: the features to train on
    :param y: the labels to predict
    :param k: how many folds to perform
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
        precision[i] = precision_score(pred, y_test)
        recall[i] = recall_score(pred, y_test)
        f1[i] = f1_score(pred, y_test)
        i += 1

    return accuracy, precision, recall, f1
