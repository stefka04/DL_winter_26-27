import numpy as np

from dl_lib.losses import functional as losses_functional
from dl_lib.perceptron import datasets
from dl_lib.perceptron import functional as perceptron_functional


def get_y_pred(dataset: list) -> np.ndarray:
    return np.array([tup[0] for tup in dataset])


def get_y_true(dataset: list) -> np.ndarray:
    return np.array([tup[1] for tup in dataset])


def train_weights(
    y_true: np.ndarray,
    x: np.ndarray,
    weights: np.ndarray,
    learning_rate=0.001,
    epochs=500,
) -> np.ndarray:
    weights_result = weights.copy()
    for epoch in np.arange(epochs):
        gradient = perceptron_functional.numerical_gradient(
            lambda w: losses_functional.mse(y_true, x * w), weights_result
        )
        weights_result = perceptron_functional.gradient_descent_step(
            weights_result, gradient, learning_rate
        )

    return weights_result


def main() -> int:
    dataset = datasets.create_dataset(6)
    y_true = get_y_true(dataset)
    x = get_y_pred(dataset)

    weights_with_seed = perceptron_functional.initialize_weights(0, 10, 1)
    print(f"Seed 42, initial w={weights_with_seed[0]:.6f}")
    weights_with_seed = train_weights(y_true, x, weights_with_seed)
    x_with_seed = x * weights_with_seed
    mse_with_seed = losses_functional.mse(y_true, x_with_seed)
    print(
        f"After 500 epochs: w={weights_with_seed[0]:.6f}, loss={mse_with_seed:.6f}, (target: w=2.0)\n"
    )

    print("Training again without a seed:")
    weights_no_seed = perceptron_functional.initialize_weights(0, 10, 1, None)
    print(f"No seed, initial w={weights_no_seed[0]:.6f}")
    weights_no_seed = train_weights(y_true, x, weights_no_seed)
    x_no_seed = x * weights_no_seed
    mse_no_seed = losses_functional.mse(y_true, x_no_seed)
    print(
        f"After 500 epochs: w={weights_no_seed[0]:.6f}, loss={mse_no_seed:.6f}, (target: w=2.0)"
    )

    return 0
