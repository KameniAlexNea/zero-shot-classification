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


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    logits, labels = eval_pred
    logits: list[np.ndarray] = sum(
        [i for j in logits for i in j if isinstance(i, list)], start=[]
    )
    labels: list[np.ndarray] = sum(
        [i for j in labels for i in j if isinstance(i, list)], start=[]
    )

    logits = np.concat([i.reshape(-1) for i in logits])
    labels = np.concat([i.reshape(-1) for i in labels])

    logits = 1 / (1 + np.exp(-logits))

    # Calculate accuracy, precision, recall and f1-score
    predictions = (logits > 0.5).astype(int)

    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    # Advanced metrics
    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

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
