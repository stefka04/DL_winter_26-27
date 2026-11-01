import numpy as np

from dl_lib.metrics import functional


def main() -> int:
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1])

    confusion_matrix = functional.confusion_matrix(y_true, y_pred)
    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}\n")
    print(f"True positives: {confusion_matrix['tp']}")
    print(f"True negatives: {confusion_matrix['tn']}")
    print(f"False positives: {confusion_matrix['fp']}")
    print(f"False negatives: {confusion_matrix['fn']}")

    return 0
