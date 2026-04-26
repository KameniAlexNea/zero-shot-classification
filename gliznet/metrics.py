import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

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


def _ndcg_at_k(ranked_relevance: np.ndarray, k: int) -> float:
    """NDCG@k for a single sample given binary relevance sorted by descending score."""
    k = min(k, len(ranked_relevance))
    if k == 0:
        return 0.0
    top_k = ranked_relevance[:k].astype(float)
    discounts = np.log2(np.arange(2, k + 2))  # log2(2), ..., log2(k+1)
    dcg = (top_k / discounts).sum()
    ideal_k = min(k, int(ranked_relevance.sum()))
    if ideal_k == 0:
        return 0.0
    idcg = (1.0 / np.log2(np.arange(2, ideal_k + 2))).sum()
    return float(dcg / idcg)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_metrics(
    eval_pred: tuple,
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict:
    """Compute ranking metrics for zero-shot classification.

    Expects dense predictions of shape (N_samples, max_labels) where padding
    positions are filled with -100.0, matching the label padding convention
    set by GliZNetForSequenceClassification.forward().

    Per-sample metrics:
        - Hit@k   : 1 if at least one positive label is in the top-k
        - NDCG@k  : normalised discounted cumulative gain at k
        - MRR     : reciprocal rank of the first positive label
        - ROC-AUC : area under the ROC curve (rank-based, threshold-free)
        - AP      : average precision (area under precision-recall curve)

    Args:
        eval_pred: (predictions, labels) — both (N_samples, max_labels).
        ks: Top-k cut-offs for Hit@k and NDCG@k.

    Returns:
        Dict of aggregated metric names → mean values across samples.
    """
    predictions, labels = eval_pred

    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    # Ensure 2-D: Trainer sometimes concatenates to (N,) if max_labels==1
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    hits = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}
    rr = []
    aucs = []
    aps = []

    for scores, gt in zip(predictions, labels):
        valid = gt != -100
        if not valid.any():
            continue
        s = scores[valid].astype(float)
        g = gt[valid].astype(float)

        n_pos = int(g.sum())
        if n_pos == 0:
            continue

        ranked = g[np.argsort(-s)]  # sort labels by descending predicted score

        for k in ks:
            hits[k].append(float(ranked[:k].any()))
            ndcgs[k].append(_ndcg_at_k(ranked, k))

        # MRR: reciprocal rank of first positive (1-indexed)
        pos_ranks = np.where(ranked > 0.5)[0]
        if len(pos_ranks) > 0:
            rr.append(1.0 / (pos_ranks[0] + 1))

        # ROC-AUC and AP require both classes present
        if len(np.unique(g)) > 1:
            probs = _sigmoid(s)
            try:
                aucs.append(roc_auc_score(g, probs))
                aps.append(average_precision_score(g, probs))
            except ValueError:
                pass

    result: dict = {}
    for k in ks:
        result[f"hit@{k}"] = float(np.mean(hits[k])) if hits[k] else 0.0
        result[f"ndcg@{k}"] = float(np.mean(ndcgs[k])) if ndcgs[k] else 0.0
    result["mrr"] = float(np.mean(rr)) if rr else 0.0
    result["roc_auc"] = float(np.mean(aucs)) if aucs else 0.0
    result["avg_precision"] = float(np.mean(aps)) if aps else 0.0
    result["num_samples"] = len(rr)

    return result
