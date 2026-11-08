import numpy as np

from dl_lib.perceptron import functional


class Perceptron:
    def __init__(
        self,
        number_of_inputs: int,
        low: float = 0.0,
        high: float = 10.0,
        seed: int | None = None,
        use_bias: bool = False,
        activation=None,
    ):
        self.number_of_inputs = number_of_inputs
        rng = np.random.default_rng(seed)
        self.weights = rng.uniform(low, high, number_of_inputs)
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = rng.uniform(low, high)
        else:
            self.bias = 0.0
        self.activation = activation

    def forward(self, n_inputs: np.ndarray) -> np.ndarray:
        return functional.forward(n_inputs, self.weights, self.bias, self.activation)
