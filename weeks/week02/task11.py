import matplotlib.pyplot as plt
import numpy as np
import torch

from dl_lib.activations import functional


def main() -> int:
    x = np.linspace(-10, 10, 30)
    sigmoid_output = functional.sigmoid(x)
    print(
        f"sigmoid range:           [{sigmoid_output.min():.4f}, {sigmoid_output.max():.4f}]"
    )
    sigmoid_output_pytorch = torch.sigmoid(torch.from_numpy(x))
    print(
        f"sigmoid range (PyTorch): [{sigmoid_output_pytorch.min():.4f}, {sigmoid_output_pytorch.max():.4f}]"
    )

    relu_output = functional.relu(x)
    print(
        f"relu range:              [{relu_output.min():.4f}, {relu_output.max():.4f}]"
    )
    relu_output_pytorch = torch.relu(torch.from_numpy(x))
    print(
        f"relu range (PyTorch):    [{relu_output_pytorch.min():.4f}, {relu_output_pytorch.max():.4f}]"
    )

    tanh_output = functional.tanh(x)
    print(
        f"tanh range:              [{tanh_output.min():.4f}, {tanh_output.max():.4f}]"
    )
    tanh_output_pytorch = torch.tanh(torch.from_numpy(x))
    print(
        f"tanh range (PyTorch):    [{tanh_output_pytorch.min():.4f}, {tanh_output_pytorch.max():.4f}]\n"
    )

    print(
        f"At x=-5: sigmoid={functional.sigmoid(np.array(-5)):.4f}, relu={functional.relu(np.array(-5)):.4f}, tanh={functional.tanh(np.array(-5)):.4f}"
    )
    print(
        f"At x=0: sigmoid={functional.sigmoid(np.array(0)):.4f}, relu={functional.relu(np.array(0)):.4f}, tanh={functional.tanh(np.array(0)):.4f}"
    )
    print(
        f"At x=5: sigmoid={functional.sigmoid(np.array(5)):.4f}, relu={functional.relu(np.array(5)):.4f}, tanh={functional.tanh(np.array(5)):.4f}"
    )

    figure, axis = plt.subplots()
    axis.plot(x, sigmoid_output, label="sigmoid")
    axis.plot(x, relu_output, label="relu")
    axis.plot(x, tanh_output, label="tanh")

    axis.set_title("sigmoid, relu, and tanh")
    axis.grid(True)
    axis.legend()
    axis.set_xlabel("x")
    axis.set_ylabel("activation(x)")
    figure.tight_layout()

    plt.show()
    # figure.savefig("w02_t11_activation_comparison.png")

    return 0
