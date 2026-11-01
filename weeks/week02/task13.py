import numpy as np

from dl_lib.losses import functional


def main() -> int:
    y_true = np.array([1.0, 0.0, 1.0])
    p = np.array([0.9, 0.2, 0.6])

    print(f"Binary cross-entropy on y_true={y_true}, p={p}:")
    print(f"{functional.binary_cross_entropy(y_true, p):.4f}\n")

    y_true = np.array([1])
    p = np.array([0.999])
    print(
        f"y_true={y_true[0]}, p={p[0]} (confidently right): loss={functional.binary_cross_entropy(y_true, p):.4f}"
    )
    p = np.array([0.001])
    print(
        f"y_true={y_true[0]}, p={p[0]} (confidently wrong): loss={functional.binary_cross_entropy(y_true, p):.4f}\n"
    )

    y_true = np.array([0, 1, 2])
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    print(
        f"Multiclass cross-entropy on y_true={y_true}: {functional.categorical_cross_entropy(y_true, p):.4f}\n"
    )

    y_true = np.array([0])
    p = np.array([[0.999, 0.001, 0]])
    print(
        f"y_true={y_true[0]}, p={p[0][0]} (confidently right): loss={functional.categorical_cross_entropy(y_true, p):.4f}"
    )
    p = np.array([[0.001, 0.999, 0]])
    print(
        f"y_true={y_true[0]}, p={p[0][0]} (confidently wrong): loss={functional.categorical_cross_entropy(y_true, p):.4f}"
    )

    return 0
