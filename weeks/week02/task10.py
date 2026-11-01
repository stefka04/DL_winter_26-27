import matplotlib.pyplot as plt
import numpy as np
import torch

from dl_lib.activations import functional


def main() -> int:
    print(f"sigmoid(-10) = {functional.sigmoid(np.array(-10)):.6f}")
    print(f"sigmoid(0) = {functional.sigmoid(np.array(0)):.6f}")
    print(f"sigmoid(10) = {functional.sigmoid(np.array(10)):.6f}")

    x = np.linspace(-10, 10, 30)
    output = functional.sigmoid(x)
    print(f"Output range: [{output.min():.6f}, {output.max():.6f}]")
    output_pytorch = torch.sigmoid(torch.from_numpy(x))
    print(
        f"Output range (PyTorch): [{output_pytorch.min():.6f}, {output_pytorch.max():.6f}]"
    )

    figure, axis = plt.subplots()
    axis.plot(x, output)
    axis.set_title("sigmoid(x)")
    axis.grid(True)
    axis.set_xlabel("x")
    axis.set_ylabel("sigmoid(x)")
    figure.tight_layout()

    plt.show()
    figure.savefig("w02_t10_sigmoid.png")

    return 0
