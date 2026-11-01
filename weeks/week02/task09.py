import matplotlib.pyplot as plt
import numpy as np

from dl_lib.metrics import functional


def main() -> int:
    y_true = np.append(np.zeros(80), np.ones(20))
    rng = np.random.default_rng(seed=42)
    probabilities = np.append(rng.uniform(0, 0.7, 80), rng.uniform(0.3, 1, 20))
    thresholds = np.round(np.arange(0.1, 1.1, 0.1), 1)
    precisions = np.zeros_like(thresholds)
    recalls = np.zeros_like(thresholds)
    index = 0

    for threshold in thresholds:
        predictions = functional.apply_threshold(probabilities, threshold)
        precisions[index] = functional.precision(y_true, predictions)
        recalls[index] = functional.recall(y_true, predictions)
        index = index + 1

    print(
        f"threshold={thresholds[1]}: precision={precisions[1]:.4f}, recall={recalls[1]:.4f}"
    )
    print(
        f"threshold={thresholds[4]}: precision={precisions[4]:.4f}, recall={recalls[4]:.4f}"
    )
    print(
        f"threshold={thresholds[7]}: precision={precisions[7]:.4f}, recall={recalls[7]:.4f}"
    )

    figure, axis = plt.subplots()
    axis.plot(thresholds, precisions, marker="o", label="precision")
    axis.plot(thresholds, recalls, marker="o", label="recall")

    axis.legend()
    axis.set_title("Precision and recall as the classification threshold rises")
    axis.set_xlabel("threshold")
    axis.set_ylabel("value")
    axis.grid(True)

    figure.tight_layout()
    plt.show()
    # figure.savefig("w02_task09_precision_recall_tradeoff.png")

    return 0
