import numpy as np
import torch

from dl_lib.activations import functional


def main() -> int:
    scores = np.array([1.0, 2.0, 3.0])
    print(f"scores: {scores}")
    softmax_output = functional.softmax(scores)
    print(f"softmax: {softmax_output}")
    print(f"sum of probabilities: {softmax_output.sum():.4f}\n")

    large_scores = np.array([1000.0, 1001.0, 1002.0])
    print(f"Large scores: {large_scores}")
    exp = np.exp(large_scores)
    naive_softmax_output = exp / np.sum(exp)
    print(f"Naive softmax (unstable): {naive_softmax_output}")
    print(f"Stable softmax: {functional.softmax(large_scores)}")
    stable_softmax_torch = torch.softmax(torch.from_numpy(large_scores), dim=-1)
    print(f"Stable softmax(PyTorch): {stable_softmax_torch}")

    return 0
