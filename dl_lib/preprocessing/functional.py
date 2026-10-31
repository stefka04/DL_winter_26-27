import numpy as np


def min_max_normalize(arr: np.ndarray, a=0, b=1) -> np.ndarray:
    min = arr.min()
    max = arr.max()
    if max == min:
        return np.zeros_like(arr)

    return ((arr - min) / (max - min)) * (b - a) + a


def standardize(arr: np.ndarray) -> np.ndarray:
    mu = arr.mean()
    sigma = arr.std()

    if sigma == 0:
        return np.zeros_like(arr)
    return (arr - mu) / sigma
