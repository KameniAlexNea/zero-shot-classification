import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    coverage_error,
    f1_score,
    hamming_loss,
    jaccard_score,
    label_ranking_average_precision_score,
    matthews_corrcoef,
    ndcg_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def hamming_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for true, pred in zip(y_true, y_pred):
        if np.sum(true) == 0 and np.sum(pred) == 0:
            scores.append(1)
        else:
            scores.append(
                np.sum(np.logical_and(true, pred)) / np.sum(np.logical_or(true, pred))
            )
    return np.mean(scores)


def compute_metrics(
    eval_pred: tuple[list[np.ndarray], list[np.ndarray]],
    activated: bool = False,
    threshold: float = 0.5,
):
    """Compute evaluation metrics including hamming score and partial match ratio."""
    logits, labels = eval_pred

    # Flatten nested lists
    while isinstance(logits[0], list):
        logits = [item for sublist in logits for item in sublist]
    while isinstance(labels[0], list):
        labels = [item for sublist in labels for item in sublist]

    # Convert to arrays
    logits = np.concatenate([j.reshape(1, -1) for j in logits])
    labels = np.concatenate([j.reshape(1, -1) for j in labels])

    # Sigmoid activation if not applied
    if not activated:
        logits = 1 / (1 + np.exp(-logits))

    predictions = (logits > threshold).astype(int)

    metrics = {}

    # Multi-label classification
    if labels.ndim > 1 and labels.shape[1] > 1:
        if labels.sum(axis=1).max() == 1:
            metrics["match_accuracy"] = (labels.argmax(axis=1) == predictions.argmax(axis=1)).mean(axis=0)
        metrics["accuracy"] = (labels == predictions).mean()
        metrics["precision"] = precision_score(
            labels, predictions, average="micro", zero_division=0
        )
        metrics["recall"] = recall_score(
            labels, predictions, average="micro", zero_division=0
        )
        metrics["f1"] = f1_score(labels, predictions, average="micro", zero_division=0)
        metrics["hamming_score"] = hamming_score(labels, predictions)

        # Jaccard and Hamming metrics
        metrics["jaccard_micro"] = jaccard_score(
            labels, predictions, average="micro", zero_division=0
        )
        metrics["jaccard_samples"] = jaccard_score(
            labels, predictions, average="samples", zero_division=0
        )
        metrics["hamming_loss"] = hamming_loss(labels, predictions)
        # Ranking-based metrics
        metrics["coverage_error"] = coverage_error(labels, logits)
        metrics["label_ranking_avg_precision"] = label_ranking_average_precision_score(
            labels, logits
        )

        # Optional ROC-AUC / Average Precision
        try:
            metrics["roc_auc"] = roc_auc_score(labels, logits, average="macro")
            metrics["avg_precision"] = average_precision_score(
                labels, logits, average="macro"
            )
        except ValueError:
            pass

    else:
        # Binary case
        metrics["accuracy"] = accuracy_score(labels, predictions)
        metrics["precision"] = precision_score(labels, predictions, zero_division=0)
        metrics["recall"] = recall_score(labels, predictions, zero_division=0)
        metrics["f1"] = f1_score(labels, predictions, zero_division=0)
        try:
            metrics["roc_auc"] = roc_auc_score(labels, logits)
            metrics["avg_precision"] = average_precision_score(labels, logits)
            metrics["matthews_corrcoef"] = matthews_corrcoef(labels, predictions)
        except ValueError:
            pass
        # Additional single-label metrics
        metrics["jaccard"] = jaccard_score(labels, predictions, zero_division=0)
        metrics["hamming_loss"] = hamming_loss(labels, predictions)

    # Diagnostic stats
    if activated:
        metrics.update(
            {
                "support": labels.size,
                "threshold": threshold,
                "num_positive": int(np.sum(labels)),
                "avg_probability": float(np.mean(logits)),
            }
        )

    return metrics


def hit_rate_at_k(
    labels_list: list[np.ndarray], logits_list: list[np.ndarray], k: int
) -> float:
    hits = []
    for labels, logits in zip(labels_list, logits_list):
        labels = labels.flatten()
        topk = np.argsort(logits.flatten())[-k:]
        true_idxs = np.where(labels == 1)[0]
        hits.append(int(len(set(topk) & set(true_idxs)) > 0))
    return np.mean(hits)


def mean_reciprocal_rank_at_k(
    labels_list: list[np.ndarray], logits_list: list[np.ndarray], k: int
) -> float:
    scores = []
    for labels, logits in zip(labels_list, logits_list):
        labels = labels.flatten()
        topk = np.argsort(logits.flatten())[-k:]
        true_idxs = np.where(labels == 1)[0]
        rank_lookup = {idx: rank for rank, idx in enumerate(topk)}
        candidate_ranks = [rank_lookup[t] for t in true_idxs if t in rank_lookup]
        scores.append(1.0 / (min(candidate_ranks) + 1) if candidate_ranks else 0.0)
    return np.mean(scores)


def precision_at_k(
    labels_list: list[np.ndarray], logits_list: list[np.ndarray], k: int
) -> float:
    precisions = []
    for lab, logit in zip(labels_list, logits_list):
        true_idxs = set(np.where(lab.flatten() == 1)[0])
        denom = min(k, len(true_idxs))
        topk = np.argsort(logit.flatten())[::-1][:k]
        if denom == 0:
            continue
        hits = sum(1 for i in topk if i in true_idxs)
        precisions.append(hits / denom)
    return np.mean(precisions) if precisions else 0.0


def compute_topk_metrics(
    eval_pred: tuple[list[np.ndarray], list[np.ndarray]],
    activated: bool = True,
    threshold: float = 0.5,
    top_k: list[int] = [1, 3, 5, 10],
):
    """Compute metrics for evaluation."""
    logits, labels = eval_pred
    while isinstance(logits[0], list):
        logits = [item for sublist in logits for item in sublist]
    while isinstance(labels[0], list):
        labels = [item for sublist in labels for item in sublist]
    logits = [j.reshape(-1) for j in logits]
    labels = [j.reshape(-1) for j in labels]
    n_labels = labels[0].shape[0] if labels else 0
    top_k = [i for i in top_k if i < n_labels]

    metrics = {}
    base = compute_metrics((logits, labels), activated=activated, threshold=threshold)
    metrics.update(base)

    # Prepare arrays for scikit-learn top-k and NDCG metrics
    y_true = np.vstack(labels)
    y_score = np.vstack(logits)
    for k in top_k:
        metrics[f"HR@{k}"] = hit_rate_at_k(labels, logits, k)
        metrics[f"MRR@{k}"] = mean_reciprocal_rank_at_k(labels, logits, k)
        metrics[f"NDCG@{k}"] = ndcg_score(y_true, y_score, k=k)
        metrics[f"Precision@{k}"] = precision_at_k(labels, logits, k)

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
