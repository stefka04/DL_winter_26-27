import numpy as np


def add_scalar_to_matrix(matrix, scalar):
    print(f"Matrix + scalar ({scalar}):")
    print(f"{matrix + scalar}\n")


def add_row_vector_to_matrix(matrix, vector):
    print(f"Matrix {matrix.shape} + row vector {vector.shape}:")
    print(f"{matrix + vector}\n")


def add_column_vector_to_matrix(matrix, column_vector):
    print(f"Matrix {matrix.shape} + column vector {column_vector.shape}:")
    print(f"{matrix + column_vector}\n")


def add_incompatible_vector_to_matrix(matrix, incompatible_vector):
    print(f"Matrix {matrix.shape} + column vector {incompatible_vector.shape}:")
    print(f"{matrix + incompatible_vector}\n")


def main() -> int:
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    add_scalar_to_matrix(matrix, 10)
    add_row_vector_to_matrix(matrix, np.array([10, 20, 30]))
    add_column_vector_to_matrix(matrix, np.array([[100], [200]]))
    add_incompatible_vector_to_matrix(matrix, np.array([100, 200]))
    return 0
