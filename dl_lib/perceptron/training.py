import numpy as np

from dl_lib.losses import functional as losses_functional
from dl_lib.perceptron import functional as perceptron_functional
from dl_lib.perceptron.perceptron import Perceptron


def train(
    model: Perceptron,
    dataset: list,
    learning_rate: float,
    epochs: int,
    loss=losses_functional.mse,
    log_every: int = 1,
) -> list:
    x = np.array([tup[0] for tup in dataset])
    y_true = np.array([tup[1] for tup in dataset])
    result = []

    for epoch in np.arange(1, epochs + 1):
        grad_weights = perceptron_functional.numerical_gradient(
            lambda w: perceptron_functional.dataset_loss(
                dataset, y_true, w, model.bias, model.activation, loss
            ),
            model.weights,
        )
        model.weights = perceptron_functional.gradient_descent_step(
            model.weights, grad_weights, learning_rate
        )

        if model.use_bias:
            bias_array = np.array([model.bias])
            grad_bias = perceptron_functional.numerical_gradient(
                lambda b: perceptron_functional.dataset_loss_given_bias(
                    dataset, y_true, model.weights, model.activation, b, loss
                ),
                bias_array,
            )
            model.bias = perceptron_functional.gradient_descent_step(
                bias_array, grad_bias, learning_rate
            )[0]

        if epoch % log_every == 0 or epoch == epochs:
            current_loss = loss(y_true, model.forward(x))
            result.append((epoch, current_loss))
    return result
