from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold


def evaluate(model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame,
             average: Any = 'binary', zero_division: Any = 'warn') -> (float, float, float, float):
    """
    Evaluates a model on a training and test set.
    :param model: The model to evaluate
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :param average: default='binary': This parameter is required for multiclass/multilabel targets.
    :param zero_division: default='warn'
    :return:
    """
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    accuracy = accuracy_score(pred, y_test)
    precision = precision_score(pred, y_test, average=average, zero_division=zero_division)
    recall = recall_score(pred, y_test, average=average, zero_division=zero_division)
    f1 = f1_score(pred, y_test, average=average, zero_division=zero_division)

    return accuracy, precision, recall, f1


def kfold_eval(model: Any, X: pd.DataFrame, y: pd.DataFrame, k: int, average: Any = 'binary',
               zero_division: Any = 'warn') -> (np.array, np.array, np.array, np.array):
    """
    Performs k-fold cross-validation on the given model.

    :param model: The model to evaluate
    :param X: the features to train on
    :param y: the labels to predict
    :param k: how many folds to perform
    :param average: default='binary': This parameter is required for multiclass/multilabel targets.
    :param zero_division: default='warn'

    :return: accuracy, precision, recall, f1
    """

    kf = KFold(n_splits=k, random_state=224)

    accuracy = np.empty(kf.n_splits)
    precision = np.empty(kf.n_splits)
    recall = np.empty(kf.n_splits)
    f1 = np.empty(kf.n_splits)

    i = 0
    for train_index, test_index in kf.split(X):
        accuracy[i], precision[i], recall[i], f1[i] = evaluate(
            model,
            X.iloc[train_index],
            X.iloc[test_index],
            y.iloc[train_index],
            y.iloc[test_index],
            average,
            zero_division,
        )
        i += 1

    return accuracy, precision, recall, f1


def pca_eval(model: Any, X: pd.DataFrame, y: pd.DataFrame, k: int, c: int, average: Any = 'binary',
             zero_division: Any = 'warn') -> (np.array, np.array, np.array, np.array):
    """
    Performs PCA evaluation on a k-fold cross-validation on the given model.

    :param model: The model to evaluate
    :param X: the features to train on
    :param y: the labels to predict
    :param k: how many folds to perform
    :param c: how many principal components
    :param average: default='binary': This parameter is required for multiclass/multilabel targets.
    :param zero_division: default='warn'

    :return: accuracy, precision, recall, f1
    """
    kf = KFold(n_splits=k, random_state=224)
    pca = PCA(n_components=c, random_state=224)
    # noinspection PyTypeChecker
    fit: PCA = pca.fit(X)

    accuracy = np.empty(kf.n_splits)
    precision = np.empty(kf.n_splits)
    recall = np.empty(kf.n_splits)
    f1 = np.empty(kf.n_splits)

    i = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train_pca = fit.transform(X_train)
        X_test_pca = fit.transform(X_test)

        model.fit(X_train_pca, y_train)
        pred = model.predict(X_test_pca)

        accuracy = accuracy_score(pred, y_test)
        precision = precision_score(pred, y_test, average=average, zero_division=zero_division)
        recall = recall_score(pred, y_test, average=average, zero_division=zero_division)
        f1 = f1_score(pred, y_test, average=average, zero_division=zero_division)
        i += 1

    return accuracy, precision, recall, f1


def rfe_eval(model: Any, X: pd.DataFrame, y: pd.DataFrame, k: int, f: int, average: Any = 'binary',
             zero_division: Any = 'warn') -> (np.array, np.array, np.array, np.array):
    """
    Performs RFE evaluation on a k-fold cross-validation on the given model.

    :param model: The model to evaluate
    :param X: the features to train on
    :param y: the labels to predict
    :param k: how many folds to perform
    :param f: how many features to select for RFE
    :param average: default='binary': This parameter is required for multiclass/multilabel targets.
    :param zero_division: default='warn'

    :return: accuracy, precision, recall, f1
    """
    kf = KFold(n_splits=k, random_state=224)
    rfe = RFE(model, n_features_to_select=f)

    accuracy = np.empty(kf.n_splits)
    precision = np.empty(kf.n_splits)
    recall = np.empty(kf.n_splits)
    f1 = np.empty(kf.n_splits)

    i = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # noinspection PyTypeChecker
        fit: RFE = rfe.fit(X_train, y_train)
        X_train_rfe = fit.transform(X_train)
        X_test_rfe = fit.transform(X_test)

        model.fit(X_train_rfe, y_train)
        pred = model.predict(X_test_rfe)

        accuracy = accuracy_score(pred, y_test)
        precision = precision_score(pred, y_test, average=average, zero_division=zero_division)
        recall = recall_score(pred, y_test, average=average, zero_division=zero_division)
        f1 = f1_score(pred, y_test, average=average, zero_division=zero_division)
        i += 1

    return accuracy, precision, recall, f1


def kbest_eval(model: Any, X: pd.DataFrame, y: pd.DataFrame, k: int, kb: int, score_func: Any = f_classif,
               average: Any = 'binary',
               zero_division: Any = 'warn') -> (np.array, np.array, np.array, np.array):
    """
    Performs k-best features evaluation on a k'-fold cross-validation on the given model.

    :param model: The model to evaluate
    :param X: the features to train on
    :param y: the labels to predict
    :param k: how many folds to perform
    :param kb: how many best features to select
    :param score_func: the scoring function for the features
    :param average: default='binary': This parameter is required for multiclass/multilabel targets.
    :param zero_division: default='warn'

    :return: accuracy, precision, recall, f1
    """
    kf = KFold(n_splits=k, random_state=224)
    kbest = SelectKBest(score_func=score_func, k=kb)

    accuracy = np.empty(kf.n_splits)
    precision = np.empty(kf.n_splits)
    recall = np.empty(kf.n_splits)
    f1 = np.empty(kf.n_splits)

    i = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # noinspection PyTypeChecker
        fit: SelectKBest = kbest.fit(X_train, y_train)
        X_train_rfe = fit.transform(X_train)
        X_test_rfe = fit.transform(X_test)

        model.fit(X_train_rfe, y_train)
        pred = model.predict(X_test_rfe)

        accuracy = accuracy_score(pred, y_test)
        precision = precision_score(pred, y_test, average=average, zero_division=zero_division)
        recall = recall_score(pred, y_test, average=average, zero_division=zero_division)
        f1 = f1_score(pred, y_test, average=average, zero_division=zero_division)
        i += 1

    return accuracy, precision, recall, f1
