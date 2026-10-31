import numpy as np


def main() -> int:
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    print("A:")
    print(f"{a}\n")
    print("B:")
    print(f"{b}\n")
    print("Elementwise:")
    print(f"{a * b}\n")
    print("Matrix multiplication:")
    print(f"{a @ b}\n")
    print("The transpose of A:")
    print(f"{a.T}\n")

    return 0
