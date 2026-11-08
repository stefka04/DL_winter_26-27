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
    print(f"Dataset: {dataset}")
    weight = perceptron_functional.initialize_weights(0, 10, 1)[0]
    print(f"w: {weight}")

    y_true = get_y_true(dataset)
    y_pred = get_y_pred(dataset)

    y_pred_new = y_pred * weight
    mse = losses_functional.mse(y_true, y_pred)
    print(f"MSE: {mse}\n")

    print("Perturbing w")
    for nudge in [0.002, 0.001, -0.001, -0.002]:
        y_pred_new = y_pred * (weight + nudge)
        mse = losses_functional.mse(y_true, y_pred_new)
        print(f"  w + ({nudge}) -> w={(weight + nudge):.6f}, MSE={mse:.6f}")
    return 0
