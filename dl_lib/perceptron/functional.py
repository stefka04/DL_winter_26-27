import numpy as np

from dl_lib.losses import functional as losses_functional


def initialize_weights(low, high, size, seed=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size)


def numerical_gradient(loss_func, params) -> np.ndarray:
    grad = np.zeros_like(params)
    eps = 0.0000001
    start_loss = loss_func(params)

    for index in np.arange(params.size):
        params.flat[index] = params.flat[index] + eps
        new_loss = loss_func(params)
        grad.flat[index] = (new_loss - start_loss) / eps
        params.flat[index] = params.flat[index] - eps

    return grad


def gradient_descent_step(params, gradient, learning_rate) -> np.ndarray:
    return params - gradient * learning_rate


def analytical_gradient(
    dataset: list,
    weights: np.ndarray,
    bias: float = 0.0,
    activation=None,
    activation_derivative=None,
    loss="mse",
) -> tuple:
    x = np.array([tup[0] for tup in dataset])
    y_true = np.array([tup[1] for tup in dataset])
    logits = np.dot(x, weights) + bias
    probabilities = activation(logits) if activation is not None else logits
    N = len(dataset)

    w_grad = np.empty(shape=(x.shape[0]))
    b_grad = np.empty(shape=(x.shape[0]))

    if loss == "mse":
        activation_der_res = (
            activation_derivative(logits) if activation_derivative is not None else 1
        )
        errors_with_der = (probabilities - y_true) * activation_der_res
        w_grad = (2 / N) * np.dot(errors_with_der, x)
        b_grad = (2 / N) * np.sum(errors_with_der)
    elif loss == "bce":
        w_grad = np.array(np.dot((probabilities - y_true), x) / N)
        b_grad = np.sum(probabilities - y_true) / N

    return (w_grad, b_grad)


def dataset_loss(
    dataset: list,
    targets: np.ndarray,
    weights: np.ndarray,
    bias: float,
    activation,
    loss=losses_functional.mse,
) -> float:
    x = np.array([tup[0] for tup in dataset])
    predictions = forward(x, weights, bias, activation)

    return loss(targets, predictions)


def dataset_loss_given_bias(
    dataset: list,
    targets: np.ndarray,
    weights: np.ndarray,
    activation,
    bias_array: np.ndarray,
    loss=losses_functional.mse,
) -> float:
    return dataset_loss(dataset, targets, weights, bias_array[0], activation, loss)


def forward(x: np.ndarray, weights: np.ndarray, bias: float = 0.0, activation=None):
    output = np.dot(x, weights) + bias
    if activation is None:
        return output
    return activation(output)
