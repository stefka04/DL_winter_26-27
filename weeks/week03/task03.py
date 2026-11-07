import numpy as np

from dl_lib.losses import functional as lf
from dl_lib.perceptron import datasets
from dl_lib.perceptron import functional as pf


def get_y_pred(dataset: list) -> np.ndarray:
    return np.array([tup[0] for tup in dataset])


def get_y_true(dataset: list) -> np.ndarray:
    return np.array([tup[1] for tup in dataset])


def main() -> int:
    dataset = datasets.create_dataset(6)
    weights = pf.initialize_weights(0, 10, 1)

    y_true = get_y_true(dataset)
    y_pred = get_y_pred(dataset)

    y_pred_new = y_pred * weights
    mse = lf.mse(y_true, y_pred_new)
    print(f"Loss before update: {mse:.6f}")

    num_grad = pf.numerical_gradient(lambda w: lf.mse(y_true, y_pred * w), weights)
    print(f"Numerical gradient: {num_grad[0]:.6f}")

    y_pred_raw_grad = y_pred * (weights - num_grad)
    mse_raw_grad = lf.mse(y_true, y_pred_raw_grad)
    print(
        f"w after subtracting the raw gradient: w={(weights - num_grad)[0]:.6f}, loss={mse_raw_grad:.6f}"
    )

    learning_rate = 0.001
    w_after_one_step = pf.gradient_descent_step(weights, num_grad, learning_rate)
    print(
        f"w after one step with learning_rate={learning_rate}: w={w_after_one_step[0]:.6f}, loss={lf.mse(y_true, y_pred * w_after_one_step):.6f}\n"
    )

    print(f"Loss before update: {mse:.6f}")
    w_after_grad_descent = weights
    for epoch in np.arange(10):
        num_grad = pf.numerical_gradient(
            lambda w: lf.mse(y_true, y_pred * w), w_after_grad_descent
        )
        w_after_grad_descent = pf.gradient_descent_step(
            w_after_grad_descent, num_grad, learning_rate
        )
        mse_after_grad_descent = lf.mse(y_true, y_pred * w_after_grad_descent)
        print(
            f"epoch {epoch + 1}: w={w_after_grad_descent[0]:.6f}, loss={mse_after_grad_descent:.6f}"
        )
    return 0
