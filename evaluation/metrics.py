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

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def flatten_nested_lists(data: list) -> list:
    """Flatten nested lists recursively."""
    while data and isinstance(data[0], list):
        data = [item for sublist in data for item in sublist]
    return data


def prepare_arrays(logits: list, labels: list) -> tuple[np.ndarray, np.ndarray]:
    """Convert lists to numpy arrays and ensure proper shape."""
    logits = flatten_nested_lists(logits)
    labels = flatten_nested_lists(labels)

    # Filter out empty predictions/labels
    valid_pairs = [(log, lab) for log, lab in zip(logits, labels) if len(log) > 0 and len(lab) > 0]
    
    if not valid_pairs:
        raise ValueError("No valid examples with labels found in the dataset")
    
    logits, labels = zip(*valid_pairs)
    
    # Flatten all predictions/labels into 1D arrays (for binary classification at label level)
    logits_flat = np.concatenate([np.asarray(log).flatten() for log in logits])
    labels_flat = np.concatenate([np.asarray(lab).flatten() for lab in labels])
    
    # Reshape to (n_samples, 1) for compatibility with metrics
    logits = logits_flat.reshape(-1, 1)
    labels = labels_flat.reshape(-1, 1)

    return logits, labels


def apply_sigmoid(logits: np.ndarray) -> np.ndarray:
    """Apply sigmoid activation to logits."""
    return 1 / (1 + np.exp(-logits))


def get_predictions(logits: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert logits to binary predictions using threshold."""
    return (logits > threshold).astype(int)


def is_multilabel(labels: np.ndarray) -> bool:
    """Check if the problem is multi-label classification."""
    return labels.ndim > 1 and labels.shape[1] > 1


def is_single_label_multiclass(labels: np.ndarray) -> bool:
    """Check if the problem is single-label multi-class (one-hot encoded)."""
    return is_multilabel(labels) and labels.sum(axis=1).max() == 1


# ============================================================================
# CUSTOM METRICS
# ============================================================================


def hamming_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Hamming score (Jaccard-like metric for multi-label)."""
    scores = []
    for true, pred in zip(y_true, y_pred):
        if np.sum(true) == 0 and np.sum(pred) == 0:
            scores.append(1.0)
        else:
            intersection = np.sum(np.logical_and(true, pred))
            union = np.sum(np.logical_or(true, pred))
            scores.append(intersection / union if union > 0 else 0.0)
    return np.mean(scores)


# ============================================================================
# CORE METRIC COMPUTATION
# ============================================================================


def compute_basic_metrics(
    labels: np.ndarray, predictions: np.ndarray, average: str = "micro"
) -> dict:
    """Compute basic classification metrics."""
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(
            labels, predictions, average=average, zero_division=0
        ),
        "recall": recall_score(labels, predictions, average=average, zero_division=0),
        "f1": f1_score(labels, predictions, average=average, zero_division=0),
    }


def compute_multilabel_metrics(
    labels: np.ndarray, predictions: np.ndarray, logits: np.ndarray
) -> dict:
    """Compute metrics specific to multi-label classification."""
    metrics = compute_basic_metrics(labels, predictions, average="micro")

    # Multi-label specific metrics
    metrics.update(
        {
            "hamming_score": hamming_score(labels, predictions),
            "jaccard_micro": jaccard_score(
                labels, predictions, average="micro", zero_division=0
            ),
            "jaccard_samples": jaccard_score(
                labels, predictions, average="samples", zero_division=0
            ),
            "hamming_loss": hamming_loss(labels, predictions),
            "coverage_error": coverage_error(labels, logits),
            "label_ranking_avg_precision": label_ranking_average_precision_score(
                labels, logits
            ),
        }
    )

    # Optional probabilistic metrics
    try:
        metrics.update(
            {
                "roc_auc": roc_auc_score(labels, logits, average="macro"),
                "avg_precision": average_precision_score(
                    labels, logits, average="macro"
                ),
            }
        )
    except ValueError:
        pass

    return metrics


def compute_binary_metrics(
    labels: np.ndarray, predictions: np.ndarray, logits: np.ndarray
) -> dict:
    """Compute metrics for binary classification."""
    metrics = compute_basic_metrics(labels, predictions, average="binary")

    # Binary specific metrics
    metrics.update(
        {
            "jaccard": jaccard_score(labels, predictions, zero_division=0),
            "hamming_loss": hamming_loss(labels, predictions),
        }
    )

    # Optional probabilistic metrics
    try:
        metrics.update(
            {
                "roc_auc": roc_auc_score(labels, logits),
                "avg_precision": average_precision_score(labels, logits),
                "matthews_corrcoef": matthews_corrcoef(labels, predictions),
            }
        )
    except ValueError:
        pass

    return metrics


def compute_diagnostic_stats(
    labels: np.ndarray, logits: np.ndarray, threshold: float
) -> dict:
    """Compute diagnostic statistics."""
    return {
        "support": labels.size,
        "threshold": threshold,
        "num_positive": int(np.sum(labels)),
        "avg_probability": float(np.mean(logits)),
    }


def compute_metrics(
    eval_pred: tuple[list[np.ndarray], list[np.ndarray]],
    activated: bool = False,
    threshold: float = 0.5,
) -> dict:
    """
    Compute comprehensive evaluation metrics for both single-label and multi-label classification.

    Args:
        eval_pred: Tuple of (logits, labels)
        activated: Whether logits are already activated (sigmoid applied)
        threshold: Threshold for binary predictions

    Returns:
        Dictionary of computed metrics
    """
    logits, labels = eval_pred

    # Prepare data
    logits, labels = prepare_arrays(logits, labels)

    # Apply sigmoid if needed
    if not activated:
        logits = apply_sigmoid(logits)

    # Get predictions
    predictions = get_predictions(logits, threshold)

    # Determine problem type and compute appropriate metrics
    if is_multilabel(labels):
        if is_single_label_multiclass(labels):
            # Single-label multi-class case (one-hot encoded)
            metrics = compute_basic_metrics(labels, predictions, average="micro")
            metrics["match_accuracy"] = (
                labels.argmax(axis=1) == logits.argmax(axis=1)
            ).mean()
        else:
            # True multi-label case
            metrics = compute_multilabel_metrics(labels, predictions, logits)
    else:
        # Binary classification
        metrics = compute_binary_metrics(labels, predictions, logits)

    # Add diagnostic stats if requested
    if activated:
        metrics.update(compute_diagnostic_stats(labels, logits, threshold))

    return metrics


# ============================================================================
# TOP-K RANKING METRICS
# ============================================================================


def hit_rate_at_k(
    labels_list: list[np.ndarray], logits_list: list[np.ndarray], k: int
) -> float:
    """Compute Hit Rate at K (whether any true label is in top-k predictions)."""
    hits = []
    for labels, logits in zip(labels_list, logits_list):
        labels_flat = labels.flatten()
        top_k_indices = np.argsort(logits.flatten())[-k:]
        true_indices = np.where(labels_flat == 1)[0]
        has_hit = len(set(top_k_indices) & set(true_indices)) > 0
        hits.append(int(has_hit))
    return np.mean(hits)


def mean_reciprocal_rank_at_k(
    labels_list: list[np.ndarray], logits_list: list[np.ndarray], k: int
) -> float:
    """Compute Mean Reciprocal Rank at K."""
    scores = []
    for labels, logits in zip(labels_list, logits_list):
        labels_flat = labels.flatten()
        top_k_indices = np.argsort(logits.flatten())[-k:]
        true_indices = np.where(labels_flat == 1)[0]

        # Create rank lookup (higher score = better rank)
        rank_lookup = {idx: k - rank for rank, idx in enumerate(top_k_indices)}
        candidate_ranks = [
            rank_lookup[idx] for idx in true_indices if idx in rank_lookup
        ]

        if candidate_ranks:
            best_rank = max(candidate_ranks)  # Best (highest) rank
            scores.append(1.0 / (k - best_rank + 1))  # Convert to position-based rank
        else:
            scores.append(0.0)

    return np.mean(scores)


def precision_at_k(
    labels_list: list[np.ndarray], logits_list: list[np.ndarray], k: int
) -> float:
    """Compute Precision at K."""
    precisions = []
    for labels, logits in zip(labels_list, logits_list):
        true_indices = set(np.where(labels.flatten() == 1)[0])
        top_k_indices = np.argsort(logits.flatten())[::-1][:k]  # Descending order

        if len(true_indices) == 0:
            continue

        hits = sum(1 for idx in top_k_indices if idx in true_indices)
        precisions.append(hits / k)

    return np.mean(precisions) if precisions else 0.0


def prepare_topk_data(
    logits: list, labels: list
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Prepare data for top-k metrics computation."""
    logits = flatten_nested_lists(logits)
    labels = flatten_nested_lists(labels)

    logits = [j.reshape(-1) for j in logits]
    labels = [j.reshape(-1) for j in labels]
    
    # Filter out empty arrays (size 0)
    valid_pairs = [(log, lab) for log, lab in zip(logits, labels) if log.size > 0 and lab.size > 0]
    
    if not valid_pairs:
        return [], []
    
    logits, labels = zip(*valid_pairs)
    logits = list(logits)
    labels = list(labels)

    return logits, labels


def compute_topk_metrics(
    eval_pred: tuple[list[np.ndarray], list[np.ndarray]],
    activated: bool = True,
    threshold: float = 0.5,
    top_k: list[int] = [1, 3, 5, 10],
) -> dict:
    """
    Compute comprehensive metrics including top-k ranking metrics.

    Args:
        eval_pred: Tuple of (logits, labels)
        activated: Whether logits are already activated
        threshold: Threshold for binary predictions
        top_k: List of k values for top-k metrics

    Returns:
        Dictionary of computed metrics including top-k metrics
    """
    logits, labels = eval_pred

    # Prepare data for top-k metrics
    logits_topk, labels_topk = prepare_topk_data(logits, labels)

    # Filter top_k values based on number of labels
    n_labels = labels_topk[0].shape[0] if labels_topk else 0
    valid_k = [k for k in top_k if k <= n_labels]

    # Compute base metrics
    metrics = compute_metrics(
        (logits, labels), activated=activated, threshold=threshold
    )

    # Compute top-k metrics
    if valid_k and n_labels > 1 and labels_topk:
        # Ensure all arrays have the same shape before vstacking
        expected_size = labels_topk[0].shape[0]
        valid_indices = [i for i, lab in enumerate(labels_topk) if lab.shape[0] == expected_size]
        
        if valid_indices:
            labels_topk_filtered = [labels_topk[i] for i in valid_indices]
            logits_topk_filtered = [logits_topk[i] for i in valid_indices]
            
            y_true = np.vstack(labels_topk_filtered)
            y_score = np.vstack(logits_topk_filtered)
            
            for k in valid_k:
                metrics[f"HR@{k}"] = hit_rate_at_k(labels_topk_filtered, logits_topk_filtered, k)
                metrics[f"MRR@{k}"] = mean_reciprocal_rank_at_k(labels_topk_filtered, logits_topk_filtered, k)
                metrics[f"Precision@{k}"] = precision_at_k(labels_topk_filtered, logits_topk_filtered, k)
                metrics[f"NDCG@{k}"] = ndcg_score(y_true, y_score, k=k)

    return metrics


# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================


def find_best_threshold(
    logits: np.ndarray, labels: np.ndarray, metric: str = "f1", n_thresholds: int = 100
) -> tuple[float, float]:
    """
    Find the best threshold for a given metric.

    Args:
        logits: Prediction probabilities
        labels: True labels
        metric: Metric to optimize ("f1", "accuracy", "precision", "recall")
        n_thresholds: Number of thresholds to test

    Returns:
        Tuple of (best_threshold, best_score)
    """
    thresholds = np.linspace(logits.min(), logits.max(), n_thresholds)
    best_score = 0
    best_threshold = 0.5

    metric_func = {
        "f1": f1_score,
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
    }[metric]

    for threshold in thresholds:
        predictions = (logits > threshold).astype(int)

        if metric == "accuracy":
            score = metric_func(labels, predictions)
        else:
            score = metric_func(labels, predictions, zero_division=0)

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


def compute_best_metrics(
    logits: list[float], labels: list[float], multi: bool = False
) -> dict:
    """
    Compute metrics with optimized threshold for best F1 score.

    Args:
        logits: Prediction scores
        labels: True labels
        multi: Whether this is multi-class classification

    Returns:
        Dictionary of metrics with optimized threshold
    """
    logits = np.array(logits)
    labels = np.array(labels)

    if multi:
        # For multi-class, use argmax instead of threshold
        predictions = logits
        threshold = None
    else:
        # Find best threshold for F1 score
        threshold, _ = find_best_threshold(logits, labels, metric="f1")
        predictions = (logits > threshold).astype(int)

    # Compute metrics
    average_mode = "weighted" if multi else "binary"

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(
            labels, predictions, zero_division=0, average=average_mode
        ),
        "recall": recall_score(
            labels, predictions, zero_division=0, average=average_mode
        ),
        "f1": f1_score(labels, predictions, zero_division=0, average=average_mode),
        "threshold": threshold,
    }

    if multi:
        print(classification_report(labels, predictions, zero_division=0))

    return metrics


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================


def main():
    """Command line interface for computing metrics."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Compute metrics for model evaluation."
    )
    parser.add_argument(
        "--file", type=str, required=True, help="Path to the evaluation file."
    )
    parser.add_argument("--multi", action="store_true", help="Use multi-class metrics.")
    args = parser.parse_args()

    # Load data
    with open(args.file, "r") as f:
        data = json.load(f)

    logits = data["detailed_results"]["predictions"]
    labels = data["detailed_results"]["true_labels"]

    # Process multi-class case
    if args.multi:
        logits = [np.argmax(i) for i in logits]
        labels = [np.argmax(i) for sublist in labels for i in sublist]
        print(f"Sample logits: {logits[:10]}")
        print(f"Sample labels: {labels[:10]}")

    # Flatten nested lists
    logits = flatten_nested_lists(logits)
    labels = flatten_nested_lists(labels)

    # Compute and display metrics
    metrics = compute_best_metrics(logits, labels, multi=args.multi)
    print("\nComputed Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
