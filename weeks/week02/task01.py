import numpy as np


def calculate_signed_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return y_pred - y_true


def calculate_absolute_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.absolute(y_pred - y_true)


def calculate_squared_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return (y_pred - y_true) ** 2


def main() -> int:
    y_true = np.array([3, 5, 7, 9])
    y_pred = np.array([2, 6, 8, 7])

    signed_errors = calculate_signed_errors(y_true, y_pred)
    absolute_errors = calculate_absolute_errors(y_true, y_pred)
    squared_errors = calculate_squared_errors(y_true, y_pred)

    print(f"y_true:          {y_true}")
    print(f"y_pred:          {y_pred}")
    print(f"errors:          {signed_errors}")
    print(f"absolute errors: {absolute_errors}")
    print(f"squared errors:  {squared_errors}\n")

    N = y_true.size
    mean_error = (1 / N) * signed_errors.sum()
    print(f"mean error: {mean_error:.4f}")
    mean_absolute_error = (1 / N) * absolute_errors.sum()
    print(f"MAE (mean absolute error): {mean_absolute_error:.4f}")
    mean_squared_error = (1 / N) * squared_errors.sum()
    print(f"MSE (mean squared error): {mean_squared_error:.4f}")
    root_mean_squared_error = np.sqrt(mean_squared_error)
    print(f"RMSE (root mean squared error): {root_mean_squared_error:.4f}")

    return 0
