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


def compute_metrics(eval_pred, activated: bool = False, threshold: float = 0.5):
    """Compute metrics for evaluation."""
    logits, labels = eval_pred
    logits: list[np.ndarray] = [i.reshape(-1) for j in logits for i in j]
    labels: list[np.ndarray] = [i.reshape(-1) for j in labels for i in j]

    logits = np.concatenate(logits)
    labels = np.concatenate(labels)

    if not activated:
        logits = 1 / (1 + np.exp(-logits))

    # Calculate accuracy, precision, recall and f1-score
    predictions = (logits > threshold).astype(int)

    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    # Advanced metrics
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    if activated:
        stats = {
            "support": len(labels),
            "threshold": threshold,
            "num_positive": np.sum(labels),
            "avg_probability": np.mean(logits),
        }
        metrics.update(stats)

    # Only calculate ROC AUC when both classes are present
    if len(np.unique(labels)) > 1:
        try:
            metrics["roc_auc"] = roc_auc_score(labels, logits)
            metrics["avg_precision"] = average_precision_score(labels, logits)
            metrics["matthews_corrcoef"] = matthews_corrcoef(labels, predictions)
        except ValueError:
            # In case of error, skip these metrics
            pass

    return metrics
