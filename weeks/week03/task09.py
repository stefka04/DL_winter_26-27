import numpy as np

from dl_lib.activations import functional as activations_functional
from dl_lib.losses import functional as losses_functional
from dl_lib.perceptron import datasets
from dl_lib.perceptron import functional as perceptron_functional
from dl_lib.perceptron.perceptron import Perceptron


def main() -> int:
    dataset = datasets.and_dataset()
    model = Perceptron(
        2,
        low=-1,
        high=1,
        use_bias=True,
        seed=42,
        activation=activations_functional.sigmoid,
    )
    print(f"AND gate (sigmoid) - weights: {model.weights}, bias: {model.bias:.6f}")
    y_true = np.array([tup[1] for tup in dataset])

    num_weight_grad = perceptron_functional.numerical_gradient(
        lambda w: perceptron_functional.dataset_loss(
            dataset,
            y_true,
            w,
            model.bias,
            model.activation,
            losses_functional.binary_cross_entropy,
        ),
        model.weights,
    )
    print(f"Numerical weight gradient: {num_weight_grad}")

    analytical_grad = perceptron_functional.analytical_gradient(
        dataset, model.weights, model.bias, model.activation, loss="bce"
    )
    analytical_weight_grad = analytical_grad[0]
    print(f"Analytical weight gradient: {analytical_weight_grad}")
    print(f"Weight gradient difference: {num_weight_grad - analytical_weight_grad}")

    num_bias_grad = perceptron_functional.numerical_gradient(
        lambda b: perceptron_functional.dataset_loss_given_bias(
            dataset,
            y_true,
            model.weights,
            model.activation,
            b,
            losses_functional.binary_cross_entropy,
        ),
        np.array([model.bias]),
    )
    print(f"Numerical bias gradient: {num_bias_grad[0]:.6f}")

    analytical_bias_grad = analytical_grad[1]
    print(f"Analytical bias gradient: {analytical_bias_grad:.6f}")
    print(f"Bias gradient difference: {num_bias_grad[0] - analytical_bias_grad}")
    return 0
