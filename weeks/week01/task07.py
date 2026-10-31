import numpy as np


def print_matrix(matrix: np.ndarray) -> None:
    print(f"Matrix:\n{matrix}\n")
    print(f"Second row: {matrix[1]}")
    print(f"Third column: {matrix[:, 2]}")
    print(f"Submatrix: {matrix[0:2, 1:3]}\n")
    print(f"Boolean mask for more than 6:\n{matrix > 6}")
    print(f"Values satisfying the mask: {matrix[matrix > 6]}")


def main() -> int:
    print_matrix(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]))
    return 0
