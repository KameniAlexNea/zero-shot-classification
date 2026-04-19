import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten(data: list) -> list:
    """Recursively flatten one level of list nesting at a time."""
    while data and isinstance(data[0], list):
        data = [item for sublist in data for item in sublist]
    return data


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _prepare(
    logits: list[np.ndarray],
    labels: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten, concatenate, and remove -100-padded positions from both arrays."""
    logits = np.concatenate([j.reshape(-1) for j in _flatten(logits)])
    labels = np.concatenate([j.reshape(-1) for j in _flatten(labels)])
    valid = labels != -100
    return logits[valid], labels[valid]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_metrics(
    eval_pred: tuple[list[np.ndarray], list[np.ndarray]],
    activated: bool = False,
    threshold: float = 0.5,
) -> dict:
    """Compute binary classification metrics for a batch of predictions.

    Args:
        eval_pred: (logits, labels) where each is a list of arrays.
        activated: If True the logits are already sigmoid probabilities.
        threshold: Decision threshold for converting probabilities to predictions.

    Returns:
        Dict of metric names → values.
    """
    logits, labels = _prepare(*eval_pred)

    probs = logits if activated else _sigmoid(logits)
    predictions = (probs > threshold).astype(int)

    metrics: dict = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "support": len(labels),
        "threshold": threshold,
        "num_positive": int(np.sum(labels)),
        "avg_probability": float(np.mean(probs)),
    }

    if len(np.unique(labels)) > 1:
        try:
            metrics["roc_auc"] = roc_auc_score(labels, probs)
            metrics["avg_precision"] = average_precision_score(labels, probs)
            metrics["matthews_corrcoef"] = matthews_corrcoef(labels, predictions)
        except ValueError:
            pass

    return metrics


def compute_best_metrics(
    logits: list[float],
    labels: list[float],
    multi: bool = False,
) -> dict:
    """Find the threshold that maximises F1 and return metrics at that threshold.

    Args:
        logits: Predicted probabilities (already sigmoid-activated).
        labels: Ground-truth binary labels.
        multi: If True, treat `logits` as pre-thresholded multi-label predictions
               and use weighted averaging.

    Returns:
        Dict of metric names → values (includes the chosen threshold).
    """
    logits = np.array(logits)
    labels = np.array(labels)

    average = "weighted" if multi else "binary"

    if multi:
        # Multi-label: logits are already binarised predictions
        predictions = logits
        threshold = None
    else:
        # Binary: search for the threshold that maximises F1
        threshold = 0.5
        best_f1 = 0.0
        for t in np.linspace(logits.min(), logits.max(), 20):
            preds = (logits > t).astype(int)
            score = f1_score(labels, preds, zero_division=0, average=average)
            if score > best_f1:
                best_f1, threshold = score, float(t)
        predictions = (logits > threshold).astype(int)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0, average=average),
        "recall": recall_score(labels, predictions, zero_division=0, average=average),
        "f1": f1_score(labels, predictions, zero_division=0, average=average),
        "threshold": threshold,
    }
