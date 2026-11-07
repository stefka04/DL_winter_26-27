import numpy as np


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
