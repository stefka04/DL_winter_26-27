import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    sigmoid_of_x = sigmoid(x)
    return sigmoid_of_x * (1 - sigmoid_of_x)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(np.zeros_like(x), x)


def tanh(x: np.ndarray) -> np.ndarray:
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    m = np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x - m)
    return exp / np.sum(exp, axis=-1, keepdims=True)
