import matplotlib.pyplot as plt
import numpy as np

from dl_lib.activations import functional as activations_functional
from dl_lib.losses import functional as losses_functional
from dl_lib.perceptron import datasets, training
from dl_lib.perceptron.perceptron import Perceptron


def print_predictions(dataset: list, model: Perceptron, logic_operation: str):
    print(f"{logic_operation} gate - predictions:")
    x = np.array([tup[0] for tup in dataset])
    y_true = np.array([tup[1] for tup in dataset])

    predictions = model.forward(x)
    for i in np.arange(y_true.size):
        print(
            f"  {logic_operation}({x[i][0]}, {x[i][1]}) = {y_true[i]} -> predicted {predictions[i]:.6f}"
        )


def main() -> int:
    epochs = 100000
    and_dataset = datasets.and_dataset()
    and_model_no_activation = Perceptron(2, low=-1, high=1, use_bias=True, seed=42)
    train_result_no_activation = training.train(
        and_model_no_activation,
        and_dataset,
        learning_rate=0.1,
        epochs=epochs,
        loss=losses_functional.mse,
        log_every=1,
    )
    print(f"Without sigmoid - final loss: {train_result_no_activation[-1][1]:.6f}")
    print_predictions(and_dataset, and_model_no_activation, logic_operation="AND")

    and_model_with_sigmoid = Perceptron(
        2,
        low=-1,
        high=1,
        use_bias=True,
        seed=42,
        activation=activations_functional.sigmoid,
    )
    train_result_with_sigmoid = training.train(
        and_model_with_sigmoid,
        and_dataset,
        learning_rate=0.1,
        epochs=epochs,
        loss=losses_functional.mse,
        log_every=1,
    )
    print(f"\nWith sigmoid    - final loss: {train_result_with_sigmoid[-1][1]:.6f}")
    print_predictions(and_dataset, and_model_with_sigmoid, logic_operation="AND")

    figure, axis = plt.subplots()
    epochs_array = np.array([tup[0] for tup in train_result_no_activation])
    all_losses_no_activation = np.array([tup[1] for tup in train_result_no_activation])
    axis.plot(epochs_array, all_losses_no_activation, label="no activation")
    all_losses_with_sigmoid = np.array([tup[1] for tup in train_result_with_sigmoid])
    axis.plot(epochs_array, all_losses_with_sigmoid, label="sigmoid")
    axis.legend()
    axis.grid(True)
    axis.set_title(f"AND gate - loss over {epochs} epochs")
    axis.set_xlabel("epoch")
    axis.set_ylabel("MSE loss")
    figure.tight_layout()

    plt.show()
    # plt.savefig("w03_task08_and_loss_curve.png")

    return 0
