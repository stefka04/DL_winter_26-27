import numpy as np


def create_dataset(n: int) -> list:
    return [(float(x), float(2 * x)) for x in np.arange(n)]
