import numpy as np

from dl_lib.metrics import functional


def main() -> int:
    imbalanced_y_true = np.append(np.zeros(990), np.ones(10))
    imbalanced_y_pred = np.zeros(1000)
    imbalanced_accuracy = functional.accuracy(imbalanced_y_true, imbalanced_y_pred)
    imbalanced_precision = functional.precision(imbalanced_y_true, imbalanced_y_pred)
    imbalanced_recall = functional.recall(imbalanced_y_true, imbalanced_y_pred)
    imbalanced_f1 = functional.f1_score(imbalanced_y_true, imbalanced_y_pred)

    print("Dataset: 990 negative, 10 positive")
    print("Model: always predicts negative, never looks at the input\n")
    print(f"accuracy:  {imbalanced_accuracy:.4f}")
    print(f"precision: {imbalanced_precision:.4f}")
    print(f"recall:    {imbalanced_recall:.4f}")
    print(f"f1:        {imbalanced_f1:.4f}")

    return 0
