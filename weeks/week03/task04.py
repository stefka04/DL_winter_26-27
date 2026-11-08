import numpy as np

from dl_lib.losses import functional as losses_functional
from dl_lib.perceptron import datasets
from dl_lib.perceptron import functional as perceptron_functional


def get_y_pred(dataset: list) -> np.ndarray:
    return np.array([tup[0] for tup in dataset])


def get_y_true(dataset: list) -> np.ndarray:
    return np.array([tup[1] for tup in dataset])


def main() -> int:
    dataset = datasets.create_dataset(6)
    weights = perceptron_functional.initialize_weights(0, 10, 1)

    y_true = get_y_true(dataset)
    y_pred = get_y_pred(dataset)

    y_pred_weighted = y_pred * weights
    mse_res = losses_functional.mse(y_true, y_pred_weighted)
    print(f"Loss: {mse_res:.6f}")

    num_grad = perceptron_functional.numerical_gradient(
        lambda w: losses_functional.mse(y_true, y_pred * w), weights
    )
    print(f"Numerical gradient: {num_grad[0]:.6f}")

    analytical_grad = perceptron_functional.analytical_gradient(dataset, weights[0])
    print(f"Analytical gradient: {analytical_grad[0]:.6f}")
    print(f"Difference: {(num_grad[0] - analytical_grad[0]):.6f}")
    return 0
