import numpy as np


def apply_threshold(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    return np.array(probabilities >= threshold, dtype=int)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, np.int64]:
    result = {}
    result["tp"] = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    result["tn"] = np.sum(np.logical_and(y_true == 0, y_pred == 0))
    result["fp"] = np.sum(np.logical_and(y_true == 0, y_pred == 1))
    result["fn"] = np.sum(np.logical_and(y_true == 1, y_pred == 0))

    return result


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    conf_matrix = confusion_matrix(y_true, y_pred)
    return (conf_matrix["tp"] + conf_matrix["tn"]) / (
        conf_matrix["tp"] + conf_matrix["tn"] + conf_matrix["fp"] + conf_matrix["fn"]
    )


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    conf_matrix = confusion_matrix(y_true, y_pred)
    if (conf_matrix["tp"] + conf_matrix["fp"]) == 0:
        return 0.0
    return conf_matrix["tp"] / (conf_matrix["tp"] + conf_matrix["fp"])


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    conf_matrix = confusion_matrix(y_true, y_pred)
    if (conf_matrix["tp"] + conf_matrix["fn"]) == 0:
        return 0.0
    return conf_matrix["tp"] / (conf_matrix["tp"] + conf_matrix["fn"])


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)

    if prec == 0 and rec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)
