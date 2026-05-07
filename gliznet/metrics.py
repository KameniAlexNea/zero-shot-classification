import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


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

    def _to_dense(arrays, pad_value: float) -> np.ndarray:
        if isinstance(arrays, np.ndarray):
            return arrays
        # Recursively collect all leaf 2-D numpy arrays from any nesting depth.
        # Needed because with eval_use_gather_object + DDP, each batch step
        # produces a list of per-rank arrays: [[gpu0, gpu1], [gpu0, gpu1], ...]
        flat: list[np.ndarray] = []

        def _collect(x) -> None:
            if isinstance(x, np.ndarray):
                flat.append(x if x.ndim == 2 else x.reshape(1, -1))
            elif isinstance(x, (list, tuple)):
                for item in x:
                    _collect(item)

        _collect(arrays)
        if not flat:
            return np.array(arrays)
        max_cols = max(a.shape[1] for a in flat)
        return np.concatenate(
            [
                np.pad(
                    a, ((0, 0), (0, max_cols - a.shape[1])), constant_values=pad_value
                )
                for a in flat
            ],
            axis=0,
        )

    predictions = _to_dense(predictions, pad_value=0.0)
    labels = _to_dense(labels, pad_value=-100)

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
            # Only compute @k when we have at least k candidates
            if len(ranked) >= k:
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
