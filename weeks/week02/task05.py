import numpy as np

from dl_lib.metrics import functional


def main() -> int:
    probabilities = np.array([0.1, 0.8, 0.4, 0.7, 0.9])
    print(f"Probabilities: {probabilities}\n")
    thresholds = np.round(np.arange(0.1, 1.0, 0.2), 1)

    for threshold in thresholds:
        predictions = functional.apply_threshold(probabilities, threshold)
        print(f"threshold={threshold}: predictions={predictions}")

    return 0
