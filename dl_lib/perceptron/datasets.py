import numpy as np


def create_dataset(n: int) -> list:
    return [(float(x), float(2 * x)) for x in np.arange(n)]


def and_dataset() -> list:
    return [(np.array([x1, x2]), x1 & x2) for x1 in [0, 1] for x2 in [0, 1]]


def or_dataset() -> list:
    return [(np.array([x1, x2]), x1 | x2) for x1 in [0, 1] for x2 in [0, 1]]


def nand_dataset() -> list:
    return [
        (np.array([x1, x2]), int(not (x1 and x2))) for x1 in [0, 1] for x2 in [0, 1]
    ]


def xor_dataset() -> list:
    return [(np.array([x1, x2]), x1 ^ x2) for x1 in [0, 1] for x2 in [0, 1]]
