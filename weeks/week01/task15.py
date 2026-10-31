import numpy as np
import torch


def main() -> int:
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(f"Numpy array:\n{arr}")
    print(f"dtype={arr.dtype}, shape={arr.shape}\n")

    tensor = torch.from_numpy(arr)
    print(f"PyTorch tensor:\n{tensor}")
    print(f"dtype={tensor.dtype}, shape={tensor.shape}, device={tensor.device}\n")

    # if torch.cuda.is_available():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Selected device: {device}")
    tensor = tensor.to(device)
    print(f"Tensor moved to device: {tensor.device}, dtype={tensor.dtype}")
    return 0
