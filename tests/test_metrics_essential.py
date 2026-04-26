"""
Pytest tests for gliznet/metrics.py.

Covers:
  - _sigmoid / _flatten helpers
  - _ndcg_at_k
  - compute_metrics  (Hit@k, NDCG@k, MRR, ROC-AUC, Avg Precision)
  - edge cases: all-padding, no positives, single sample, perfect ranking
"""

import numpy as np
import pytest

from gliznet.metrics import (
    _flatten,
    _ndcg_at_k,
    _sigmoid,
    compute_metrics,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_eval_pred(scores_list, labels_list):
    """Build a dense (N, max_labels) eval_pred tuple with -100 padding."""
    max_k = max(len(s) for s in scores_list)
    scores_arr = np.full((len(scores_list), max_k), -100.0)
    labels_arr = np.full((len(labels_list), max_k), -100.0)
    for i, (s, l) in enumerate(zip(scores_list, labels_list)):
        scores_arr[i, : len(s)] = s
        labels_arr[i, : len(l)] = l
    return scores_arr, labels_arr


# ──────────────────────────────────────────────────────────────────────────────
# _flatten
# ──────────────────────────────────────────────────────────────────────────────


class TestFlatten:
    def test_single_level(self):
        assert _flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]

    def test_already_flat(self):
        assert _flatten([1, 2, 3]) == [1, 2, 3]

    def test_empty(self):
        assert _flatten([]) == []

    def test_double_level(self):
        assert _flatten([[[1, 2]], [[3, 4]]]) == [1, 2, 3, 4]


# ──────────────────────────────────────────────────────────────────────────────
# _sigmoid
# ──────────────────────────────────────────────────────────────────────────────


class TestSigmoid:
    def test_zero_maps_to_half(self):
        assert _sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)

    def test_large_positive_approaches_one(self):
        assert _sigmoid(np.array([100.0]))[0] > 0.9999

    def test_large_negative_approaches_zero(self):
        assert _sigmoid(np.array([-100.0]))[0] < 1e-4

    def test_monotone_increasing(self):
        xs = np.linspace(-3, 3, 20)
        ys = _sigmoid(xs)
        assert (np.diff(ys) > 0).all()

    def test_matches_formula(self):
        xs = np.array([-2.0, 0.0, 1.5])
        expected = 1.0 / (1.0 + np.exp(-xs))
        np.testing.assert_allclose(_sigmoid(xs), expected)


# ──────────────────────────────────────────────────────────────────────────────
# _ndcg_at_k
# ──────────────────────────────────────────────────────────────────────────────


class TestNdcgAtK:
    def test_perfect_ranking(self):
        # First item is the only positive → ideal case
        ranked = np.array([1, 0, 0, 0])
        assert _ndcg_at_k(ranked, k=1) == pytest.approx(1.0)

    def test_positive_not_in_top_k(self):
        ranked = np.array([0, 0, 1])
        assert _ndcg_at_k(ranked, k=1) == pytest.approx(0.0)

    def test_all_positives_perfect(self):
        ranked = np.array([1, 1, 1])
        assert _ndcg_at_k(ranked, k=3) == pytest.approx(1.0)

    def test_no_positives_returns_zero(self):
        ranked = np.array([0, 0, 0])
        assert _ndcg_at_k(ranked, k=3) == pytest.approx(0.0)

    def test_k_larger_than_array(self):
        ranked = np.array([1, 0])
        # Should not raise; clips to len(ranked)
        val = _ndcg_at_k(ranked, k=100)
        assert 0.0 <= val <= 1.0

    def test_partial_ranking(self):
        # Second positive is rank-2 (0-indexed rank 1)
        ranked = np.array([1, 1, 0, 0])
        val = _ndcg_at_k(ranked, k=4)
        assert 0.0 < val <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# compute_metrics
# ──────────────────────────────────────────────────────────────────────────────


class TestComputeMetrics:
    """Tests against the actual gliznet/metrics.py compute_metrics."""

    def test_output_keys_present(self):
        scores = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        labels = [[1, 0, 0], [0, 1, 0]]
        metrics = compute_metrics(_make_eval_pred(scores, labels))
        for key in ("hit@1", "hit@3", "hit@5", "ndcg@1", "ndcg@3", "ndcg@5", "mrr"):
            assert key in metrics, f"missing key: {key}"

    def test_perfect_ranking(self):
        """Positive always ranked first → Hit@1=MRR=NDCG@1=1."""
        scores = [[5.0, -1.0, -2.0]] * 4
        labels = [[1, 0, 0]] * 4
        m = compute_metrics(_make_eval_pred(scores, labels))
        assert m["hit@1"] == pytest.approx(1.0)
        assert m["mrr"] == pytest.approx(1.0)
        assert m["ndcg@1"] == pytest.approx(1.0)

    def test_worst_ranking(self):
        """Positive always ranked last → Hit@1=0, MRR minimal."""
        scores = [[-2.0, -1.0, 5.0]] * 4
        labels = [[1, 0, 0]] * 4
        m = compute_metrics(_make_eval_pred(scores, labels))
        assert m["hit@1"] == pytest.approx(0.0)
        assert m["mrr"] == pytest.approx(1 / 3)  # positive at rank 3

    def test_hit_at_k_increases_with_k(self):
        scores = [[3.0, 2.0, 1.0, 0.0]]
        labels = [[0, 0, 0, 1]]          # positive at rank 4
        m = compute_metrics(_make_eval_pred(scores, labels), ks=(1, 3, 5))
        assert m["hit@1"] == 0.0
        assert m["hit@3"] == 0.0
        assert m["hit@5"] == 1.0

    def test_padding_ignored(self):
        """Samples padded with -100 should be skipped without error."""
        scores = np.array([[5.0, -1.0, -100.0], [5.0, -1.0, -100.0]])
        labels = np.array([[1, 0, -100], [1, 0, -100]])
        m = compute_metrics((scores, labels))
        assert m["hit@1"] == pytest.approx(1.0)

    def test_all_padding_skipped(self):
        """A fully-padded sample contributes nothing."""
        scores = np.array([[-100.0, -100.0]])
        labels = np.array([[-100, -100]])
        m = compute_metrics((scores, labels))
        assert m["num_samples"] == 0

    def test_no_positives_skipped(self):
        """Samples without any positive label are skipped."""
        scores = np.array([[1.0, 0.5]])
        labels = np.array([[0, 0]])
        m = compute_metrics((scores, labels))
        assert m["num_samples"] == 0

    def test_multiple_positives(self):
        """Multiple positives per sample — MRR uses the first positive in ranked order."""
        scores = [[3.0, 2.0, 1.0]]
        labels = [[0, 1, 1]]           # two positives; top-1 is negative
        m = compute_metrics(_make_eval_pred(scores, labels), ks=(1, 2, 3))
        assert m["hit@1"] == 0.0
        assert m["hit@2"] == pytest.approx(1.0)  # second rank is positive
        assert m["mrr"] == pytest.approx(1 / 2)

    def test_roc_auc_and_ap_bounded(self):
        scores = [[2.0, 1.0, -1.0, -2.0]] * 10
        labels = [[1, 1, 0, 0]] * 10
        m = compute_metrics(_make_eval_pred(scores, labels))
        assert 0.0 <= m["roc_auc"] <= 1.0
        assert 0.0 <= m["avg_precision"] <= 1.0

    def test_single_sample(self):
        scores = [[1.0, 0.0]]
        labels = [[1, 0]]
        m = compute_metrics(_make_eval_pred(scores, labels))
        assert m["hit@1"] == pytest.approx(1.0)

    def test_ndcg_perfect_vs_reversed(self):
        """Perfect ranking should have higher NDCG than reversed."""
        perfect_scores = [[3.0, 2.0, 1.0, 0.0]]
        reversed_scores = [[0.0, 1.0, 2.0, 3.0]]
        labels = [[1, 1, 0, 0]]
        m_perfect = compute_metrics(_make_eval_pred(perfect_scores, labels), ks=(1, 3, 4))
        m_reversed = compute_metrics(_make_eval_pred(reversed_scores, labels), ks=(1, 3, 4))
        assert m_perfect["ndcg@4"] > m_reversed["ndcg@4"]

    @pytest.mark.parametrize("k", [1, 3, 5])
    def test_hit_at_k_bounds(self, k):
        scores = [[2.0, 1.0, 0.0]] * 5
        labels = [[1, 0, 0]] * 5
        m = compute_metrics(_make_eval_pred(scores, labels), ks=(k,))
        assert 0.0 <= m[f"hit@{k}"] <= 1.0

    @pytest.mark.parametrize("k", [1, 3, 5])
    def test_ndcg_at_k_bounds(self, k):
        scores = [[2.0, 1.0, 0.0]] * 5
        labels = [[1, 0, 0]] * 5
        m = compute_metrics(_make_eval_pred(scores, labels), ks=(k,))
        assert 0.0 <= m[f"ndcg@{k}"] <= 1.0




