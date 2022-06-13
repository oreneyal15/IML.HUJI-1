from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    # raise NotImplementedError()
    idx_split = np.array_split(list(range(X.shape[0])), cv, axis=0)
    validations = [(X[v], y[v]) for v in idx_split]
    validation_score, train_score = [], []
    for i in range(cv):
        train_X, train_y = np.delete(X, idx_split[i], axis=0), np.delete(y, idx_split[i], axis=0)
        model = estimator.fit(train_X, train_y)
        val_pred = model.predict(validations[i][0])
        train_pred = model.predict(train_X)
        validation_score.append(scoring(validations[i][1], val_pred))
        train_score.append(scoring(train_y, train_pred))
    return float(np.mean(train_score)), float(np.mean(validation_score))
