import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(eval_pred, activated: bool = False, threshold: float = 0.5):
    """Compute metrics for evaluation."""
    logits, labels = eval_pred
    logits: list[np.ndarray] = (
        [i.reshape(-1) for j in logits for i in j]
        if isinstance(logits[0], list)
        else [i.reshape(-1) for i in logits]
    )
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


def compute_best_metrics(logits: list[float], labels: list[float], multi: bool = False):
    """Compute best metrics for a given threshold."""
    logits = np.array(logits)
    labels = np.array(labels)

    threshold = None

    if not multi:
        # Find the best threshold for optimizing F1 score
        thresholds = np.linspace(min(logits), max(logits), 20)
        best_f1 = 0
        threshold = 0.5  # default threshold

        for t in thresholds:
            preds = (logits > t).astype(int)
            current_f1 = f1_score(
                labels,
                preds,
                zero_division=0,
                average="weighted" if multi else "binary",
            )
            if current_f1 > best_f1:
                best_f1 = current_f1
                threshold = t

        predictions = (logits > threshold).astype(int)
    else:
        predictions = logits

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(
        labels, predictions, zero_division=0, average="weighted" if multi else "binary"
    )
    recall = recall_score(
        labels, predictions, zero_division=0, average="weighted" if multi else "binary"
    )
    f1 = f1_score(
        labels, predictions, zero_division=0, average="weighted" if multi else "binary"
    )

    if multi:
        print(classification_report(labels, predictions, zero_division=0))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold,
    }


if __name__ == "__main__":
    # Example usage
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Compute metrics for model evaluation."
    )
    parser.add_argument("--file", type=str, help="Path to the evaluation file.")
    parser.add_argument("--multi", action="store_true", help="Use multi-label metrics.")
    args = parser.parse_args()

    data = json.load(open(args.file, "r"))
    logits = data["detailed_results"]["predictions"]
    labels = data["detailed_results"]["true_labels"]
    if args.multi:
        logits = [np.argmax(i) for i in logits]
        labels = [np.argmax(i) for sublist in labels for i in sublist]
        print(logits[:10], labels[:10])  # Print first 10 for debugging
    if isinstance(logits[0], list):
        logits = [item for sublist in logits for item in sublist]
    if isinstance(logits[0], list):
        logits = [item for sublist in logits for item in sublist]
    if isinstance(labels[0], list):
        labels = [item for sublist in labels for item in sublist]
    if isinstance(labels[0], list):
        labels = [item for sublist in labels for item in sublist]

    metrics = compute_best_metrics(logits, labels, multi=args.multi)
    print(metrics)
