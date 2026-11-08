import numpy as np

from dl_lib.losses import functional as losses_functional
from dl_lib.perceptron import datasets, training
from dl_lib.perceptron.perceptron import Perceptron


def print_training_result(dataset: list, model: Perceptron, epochs: int):
    train_result = training.train(
        model,
        dataset,
        learning_rate=0.1,
        epochs=epochs,
        loss=losses_functional.mse,
        log_every=10000,
    )
    for epoch, loss in train_result:
        print(f"  epoch {epoch}: loss={loss:.6f}")


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
    and_model = Perceptron(2, low=-1, high=1, use_bias=True, seed=42)
    print(f"AND gate - initial weights: {and_model.weights}, bias:{and_model.bias:.6f}")
    print_training_result(and_dataset, and_model, epochs)
    print(f"AND gate - final weights: {and_model.weights}, bias:{and_model.bias:.6f}")
    print_predictions(and_dataset, and_model, logic_operation="AND")

    or_dataset = datasets.or_dataset()
    or_model = Perceptron(2, low=-1, high=1, use_bias=True, seed=42)
    print(f"\nOR gate - initial weights: {or_model.weights}, bias:{or_model.bias:.6f}")
    print_training_result(or_dataset, or_model, epochs)
    print(f"OR gate - final weights: {or_model.weights}, bias:{or_model.bias:.6f}")
    print_predictions(or_dataset, or_model, logic_operation="OR")

    return 0
