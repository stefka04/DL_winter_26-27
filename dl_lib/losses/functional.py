import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    absolute_errors = np.absolute(y_pred - y_true)
    return (1 / y_true.size) * absolute_errors.sum()


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    return (1 / y_true.size) * (((y_pred - y_true) ** 2).sum())


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    return np.sqrt(mse(y_true, y_pred))


def binary_cross_entropy(
    y_true: np.ndarray, pred_probabilities: np.ndarray
) -> np.float64:
    epsilon = 0.0000001
    pred_probabilities = np.clip(pred_probabilities, epsilon, 1 - epsilon)
    return -np.mean(
        y_true * np.log(pred_probabilities)
        + (1 - y_true) * np.log(1 - pred_probabilities)
    )


def categorical_cross_entropy(
    y_true: np.ndarray, pred_probabilities: np.ndarray
) -> np.float64:
    epsilon = 0.0000001
    pred_probabilities = np.clip(pred_probabilities, epsilon, 1 - epsilon)
    return -np.mean(np.log(pred_probabilities[np.arange(y_true.size), y_true]))
