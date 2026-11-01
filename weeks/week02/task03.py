import numpy as np
import torch

from dl_lib.losses import functional


def main() -> int:
    y_true = np.array([3.0, 5.0, 7.0, 9.0])
    y_pred = np.array([2.0, 6.0, 8.0, 7.0])

    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    print(f"MSE: {functional.mse(y_true, y_pred):.4f}")
    mse_torch = torch.nn.functional.mse_loss(
        torch.from_numpy(y_pred), torch.from_numpy(y_true)
    )
    print(f"MSE (PyTorch): {mse_torch:.4f}\n")

    y_pred = np.array([2.0, 6.0, 8.0, 100.0])
    print(f"y_pred with one extreme outlier: {y_pred}")
    print(f"MAE with outlier: {functional.mae(y_true, y_pred):.4f}")
    print(f"MSE with outlier: {functional.mse(y_true, y_pred):.4f}")

    return 0
