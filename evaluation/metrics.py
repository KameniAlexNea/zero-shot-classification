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


def compute_metrics(
    eval_pred: tuple[list[np.ndarray], list[np.ndarray]],
    activated: bool = False,
    threshold: float = 0.5,
):
    """Compute metrics for evaluation."""
    logits, labels = eval_pred
    while isinstance(logits[0], list):
        logits = [item for sublist in logits for item in sublist]
    while isinstance(labels[0], list):
        labels = [item for sublist in labels for item in sublist]
    logits = [j.reshape(-1) for j in logits]
    labels = [j.reshape(-1) for j in labels]

    logits = np.concatenate(logits)
    labels = np.concatenate(labels)
    labels = labels[labels != -100]  # Remove -100 labels if present

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


def hit_rate_at_k(labels_list: list[np.ndarray], logits_list: list[np.ndarray], k: int) -> float:
    topk = [np.argsort(lb.flatten())[-k:] for lb in logits_list]
    hits = []
    for lab, inds in zip(labels_list, topk):
        true_idxs = np.where(lab.flatten() == 1)[0]
        hits.append(int(bool(true_idxs.size and any(i in inds for i in true_idxs))))
    return np.mean(hits)


def mean_reciprocal_rank_at_k(labels_list: list[np.ndarray], logits_list: list[np.ndarray], k: int) -> float:
    topk = [np.argsort(lb.flatten())[-k:] for lb in logits_list]
    scores = []
    for lab, inds in zip(labels_list, topk):
        true_idxs = np.where(lab.flatten() == 1)[0]
        if not true_idxs.size:
            continue
        # pick best (smallest) rank among true labels
        ranks = [np.where(inds == t)[0][0] for t in true_idxs if t in inds]
        scores.append(1.0 / (min(ranks) + 1) if ranks else 0.0)
    return np.mean(scores) if scores else 0.0


def ndcg_at_k(labels_list: list[np.ndarray], logits_list: list[np.ndarray], k: int) -> float:
    scores = []
    for lab, lb in zip(labels_list, logits_list):
        flat_lab = lab.flatten()
        if not flat_lab.sum():
            continue
        # ideal DCG
        ideal_order = np.argsort(flat_lab)[::-1]
        gains = flat_lab[ideal_order]
        discounts = np.log2(np.arange(2, gains.size + 2))
        ideal_dcg = (gains / discounts).sum()
        # actual DCG@k
        pred_order = np.argsort(lb.flatten())[::-1][:k]
        actual_gains = flat_lab[pred_order]
        actual_dcg = (actual_gains / np.log2(np.arange(2, k + 2))).sum()
        scores.append(actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0)
    return np.mean(scores) if scores else 0.0


def compute_topk_metrics(
    eval_pred: tuple[list[np.ndarray], list[np.ndarray]],
    threshold: float = 0.5,
    top_k: list[int] = [1, 3, 5, 10],
):
    """Compute metrics for evaluation."""
    logits, labels = eval_pred
    while isinstance(logits[0], list) and not isinstance(logits[0][0], list):
        logits = [item for sublist in logits for item in sublist]
    while isinstance(labels[0], list) and not isinstance(labels[0][0], list):
        labels = [item for sublist in labels for item in sublist]
    logits = [j.reshape(1, -1) for j in logits]
    labels = [j.reshape(1, -1) for j in labels]
    n_labels = max(i.shape[1] for i in labels)
    top_k = [i for i in top_k if i < n_labels]

    metrics = {}
    base = compute_metrics((logits, labels), activated=True, threshold=threshold)
    metrics.update(base)

    labels = [j[j != -100].reshape(1, -1) for j in labels]

    for k in top_k:
        metrics[f"HR@{k}"] = hit_rate_at_k(labels, logits, k)
        metrics[f"MRR@{k}"] = mean_reciprocal_rank_at_k(labels, logits, k)
        metrics[f"NDCG@{k}"] = ndcg_at_k(labels, logits, k)

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

    while isinstance(logits[0], list):
        logits = [item for sublist in logits for item in sublist]
    while isinstance(labels[0], list):
        labels = [item for sublist in labels for item in sublist]

    metrics = compute_best_metrics(logits, labels, multi=args.multi)
    print(metrics)
