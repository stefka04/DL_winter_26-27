import numpy as np

from dl_lib.preprocessing.functional import min_max_normalize, standardize


def main() -> int:
    arr = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    normalized_arr = min_max_normalize(arr)
    standardized_arr = standardize(arr)
    print(f"Original: {arr}")
    print(f"Min-max: {normalized_arr}")
    print(f"Standardized: {standardized_arr}\n")

    print(f"Min-max range: [{normalized_arr.min():.4f}, {normalized_arr.max():.4f}]")
    print(
        f"Standardized mean/std: {standardized_arr.mean():.4f} / {standardized_arr.std():.4f}"
    )
    return 0
