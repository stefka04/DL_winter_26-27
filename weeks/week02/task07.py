import numpy as np

from dl_lib.metrics import functional


def main() -> int:
    balanced_y_true = np.array([1, 0, 1, 0, 1])
    balanced_y_pred = np.array([1, 0, 0, 1, 1])

    balanced_accuracy = functional.accuracy(balanced_y_true, balanced_y_pred)
    balanced_precision = functional.precision(balanced_y_true, balanced_y_pred)
    balanced_recall = functional.recall(balanced_y_true, balanced_y_pred)
    balanced_f1 = functional.f1_score(balanced_y_true, balanced_y_pred)

    print("Balanced dataset:\n")
    print(f"  accuracy:  {balanced_accuracy:.4f}")
    print(f"  precision: {balanced_precision:.4f}")
    print(f"  recall:    {balanced_recall:.4f}")
    print(f"  f1:        {balanced_f1:.4f}")

    imbalanced_y_true = np.append(np.zeros(990), np.ones(10))
    imbalanced_y_pred = np.append(np.zeros(993), np.ones(7))
    imbalanced_accuracy = functional.accuracy(imbalanced_y_true, imbalanced_y_pred)
    imbalanced_precision = functional.precision(imbalanced_y_true, imbalanced_y_pred)
    imbalanced_recall = functional.recall(imbalanced_y_true, imbalanced_y_pred)
    imbalanced_f1 = functional.f1_score(imbalanced_y_true, imbalanced_y_pred)

    print("Imbalanced dataset:\n")
    print(f"  accuracy:  {imbalanced_accuracy:.4f}")
    print(f"  precision: {imbalanced_precision:.4f}")
    print(f"  recall:    {imbalanced_recall:.4f}")
    print(f"  f1:        {imbalanced_f1:.4f}")

    # imbalanced_y_pred = np.zeros(1000)
    # imbalanced_accuracy = functional.accuracy(imbalanced_y_true, imbalanced_y_pred)
    # print(f"  accuracy (if a model predicts all observations as negative):  {imbalanced_accuracy:.4f}")

    return 0
