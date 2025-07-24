"""
Unit tests for the metrics module using unittest framework.
Tests all functions in evaluation/metrics.py for multi-label classification.
"""

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

# Add the parent directory to the path to import the evaluation module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.metrics import (
    apply_sigmoid,
    compute_basic_metrics,
    compute_best_metrics,
    compute_binary_metrics,
    compute_diagnostic_stats,
    compute_metrics,
    compute_multilabel_metrics,
    compute_topk_metrics,
    find_best_threshold,
    flatten_nested_lists,
    get_predictions,
    hamming_score,
    hit_rate_at_k,
    is_multilabel,
    is_single_label_multiclass,
    mean_reciprocal_rank_at_k,
    precision_at_k,
    prepare_arrays,
    prepare_topk_data,
)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions for data preparation and basic operations."""

    def test_flatten_nested_lists_single_level(self):
        """Test flattening single level nested lists."""
        data = [[1, 2], [3, 4], [5, 6]]
        result = flatten_nested_lists(data)
        expected = [1, 2, 3, 4, 5, 6]
        self.assertEqual(result, expected)

    def test_flatten_nested_lists_double_level(self):
        """Test flattening double level nested lists."""
        data = [[[1, 2]], [[3, 4]], [[5, 6]]]
        result = flatten_nested_lists(data)
        expected = [1, 2, 3, 4, 5, 6]
        self.assertEqual(result, expected)

    def test_flatten_nested_lists_empty(self):
        """Test flattening empty list."""
        data = []
        result = flatten_nested_lists(data)
        self.assertEqual(result, [])

    def test_flatten_nested_lists_already_flat(self):
        """Test flattening already flat list."""
        data = [1, 2, 3, 4]
        result = flatten_nested_lists(data)
        self.assertEqual(result, [1, 2, 3, 4])

    def test_prepare_arrays_basic(self):
        """Test basic array preparation."""
        logits = [np.array([0.1, 0.9]), np.array([0.8, 0.2])]
        labels = [np.array([0, 1]), np.array([1, 0])]

        result_logits, result_labels = prepare_arrays(logits, labels)

        self.assertEqual(result_logits.shape, (2, 2))
        self.assertEqual(result_labels.shape, (2, 2))
        np.testing.assert_array_equal(result_logits[0], [0.1, 0.9])
        np.testing.assert_array_equal(result_labels[0], [0, 1])

    def test_apply_sigmoid_known_values(self):
        """Test sigmoid activation function with known values."""
        logits = np.array([0, 1, -1, 10, -10])
        result = apply_sigmoid(logits)

        # Test known values
        self.assertAlmostEqual(result[0], 0.5, places=6)  # sigmoid(0) = 0.5
        self.assertGreater(result[3], 0.99)  # sigmoid(10) ≈ 1
        self.assertLess(result[4], 0.01)  # sigmoid(-10) ≈ 0

    def test_apply_sigmoid_monotonic(self):
        """Test that sigmoid is monotonically increasing."""
        logits = np.array([-2, -1, 0, 1, 2])
        result = apply_sigmoid(logits)

        # Check monotonicity
        for i in range(len(result) - 1):
            self.assertLess(result[i], result[i + 1])

    def test_get_predictions_default_threshold(self):
        """Test prediction generation with default threshold."""
        logits = np.array([0.3, 0.7, 0.1, 0.9])
        predictions = get_predictions(logits)
        expected = np.array([0, 1, 0, 1])
        np.testing.assert_array_equal(predictions, expected)

    def test_get_predictions_custom_threshold(self):
        """Test prediction generation with custom threshold."""
        logits = np.array([0.3, 0.7, 0.1, 0.9])
        predictions = get_predictions(logits, threshold=0.8)
        expected = np.array([0, 0, 0, 1])
        np.testing.assert_array_equal(predictions, expected)

    def test_is_multilabel_true(self):
        """Test multilabel detection for true multilabel data."""
        labels = np.array([[1, 0, 1], [0, 1, 1]])
        self.assertTrue(is_multilabel(labels))

    def test_is_multilabel_false_1d(self):
        """Test multilabel detection for 1D data."""
        labels = np.array([1, 0, 1, 0])
        self.assertFalse(is_multilabel(labels))

    def test_is_multilabel_false_single_column(self):
        """Test multilabel detection for single column 2D data."""
        labels = np.array([[1], [0], [1]])
        self.assertFalse(is_multilabel(labels))

    def test_is_single_label_multiclass_true(self):
        """Test single-label multiclass detection for one-hot encoded data."""
        labels = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertTrue(is_single_label_multiclass(labels))

    def test_is_single_label_multiclass_false(self):
        """Test single-label multiclass detection for true multilabel data."""
        labels = np.array([[1, 1, 0], [0, 1, 1]])
        self.assertFalse(is_single_label_multiclass(labels))


class TestCustomMetrics(unittest.TestCase):
    """Test custom metric implementations."""

    def test_hamming_score_perfect_match(self):
        """Test Hamming score with perfect predictions."""
        y_true = np.array([[1, 0, 1], [0, 1, 0]])
        y_pred = np.array([[1, 0, 1], [0, 1, 0]])
        score = hamming_score(y_true, y_pred)
        self.assertEqual(score, 1.0)

    def test_hamming_score_no_match(self):
        """Test Hamming score with no correct predictions."""
        y_true = np.array([[1, 0, 1], [0, 1, 0]])
        y_pred = np.array([[0, 1, 0], [1, 0, 1]])
        score = hamming_score(y_true, y_pred)
        self.assertEqual(score, 0.0)

    def test_hamming_score_partial_match(self):
        """Test Hamming score with partial matches."""
        y_true = np.array([[1, 0, 1], [0, 1, 0]])
        y_pred = np.array([[1, 0, 0], [0, 1, 1]])
        score = hamming_score(y_true, y_pred)
        # First sample: intersection=1, union=2, score=0.5
        # Second sample: intersection=1, union=2, score=0.5
        # Average = 0.5
        self.assertAlmostEqual(score, 0.5, places=6)

    def test_hamming_score_all_zeros(self):
        """Test Hamming score when both true and pred are all zeros."""
        y_true = np.array([[0, 0, 0], [0, 0, 0]])
        y_pred = np.array([[0, 0, 0], [0, 0, 0]])
        score = hamming_score(y_true, y_pred)
        self.assertEqual(score, 1.0)

    def test_hamming_score_mixed_cases(self):
        """Test Hamming score with mixed zero and non-zero cases."""
        y_true = np.array([[0, 0, 0], [1, 0, 1]])
        y_pred = np.array([[0, 0, 0], [1, 1, 0]])
        score = hamming_score(y_true, y_pred)
        # First sample: both all zeros, score=1.0
        # Second sample: intersection=1, union=3, score=1/3
        # Average = (1.0 + 1/3) / 2 = 2/3
        expected = (1.0 + 1 / 3) / 2
        self.assertAlmostEqual(score, expected, places=6)


class TestMetricComputation(unittest.TestCase):
    """Test metric computation functions."""

    def test_compute_basic_metrics_perfect(self):
        """Test basic metrics with perfect predictions."""
        labels = np.array([[1, 0], [0, 1]])
        predictions = np.array([[1, 0], [0, 1]])

        metrics = compute_basic_metrics(labels, predictions)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)

    def test_compute_basic_metrics_all_wrong(self):
        """Test basic metrics with all wrong predictions."""
        labels = np.array([[1, 0], [0, 1]])
        predictions = np.array([[0, 1], [1, 0]])

        metrics = compute_basic_metrics(labels, predictions)

        self.assertEqual(metrics["accuracy"], 0.0)
        # Other metrics might be 0 due to zero_division=0

    def test_compute_basic_metrics_bounds(self):
        """Test that basic metrics are within expected bounds."""
        labels = np.array([[1, 0], [0, 1], [1, 1]])
        predictions = np.array([[1, 0], [1, 1], [0, 0]])

        metrics = compute_basic_metrics(labels, predictions, average="micro")

        # All metrics should be between 0 and 1
        for metric_name, value in metrics.items():
            self.assertGreaterEqual(value, 0, f"{metric_name} should be >= 0")
            self.assertLessEqual(value, 1, f"{metric_name} should be <= 1")

    def test_compute_multilabel_metrics_structure(self):
        """Test multilabel-specific metrics computation structure."""
        labels = np.array([[1, 0, 1], [0, 1, 0]])
        predictions = np.array([[1, 0, 0], [0, 1, 1]])
        logits = np.array([[0.8, 0.2, 0.3], [0.1, 0.9, 0.7]])

        metrics = compute_multilabel_metrics(labels, predictions, logits)

        # Check that all expected metrics are present
        expected_keys = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "hamming_score",
            "jaccard_micro",
            "jaccard_samples",
            "hamming_loss",
            "coverage_error",
            "label_ranking_avg_precision",
        ]
        for key in expected_keys:
            self.assertIn(key, metrics)

        # Check bounds for most metrics
        bounded_metrics = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "hamming_score",
            "jaccard_micro",
            "jaccard_samples",
            "hamming_loss",
        ]
        for metric_name in bounded_metrics:
            if metric_name in metrics:
                self.assertGreaterEqual(metrics[metric_name], 0)
                self.assertLessEqual(metrics[metric_name], 1)

    def test_compute_binary_metrics_structure(self):
        """Test binary classification metrics computation structure."""
        labels = np.array([1, 0, 1, 0])
        predictions = np.array([1, 0, 0, 0])
        logits = np.array([0.8, 0.2, 0.3, 0.1])

        metrics = compute_binary_metrics(labels, predictions, logits)

        # Check that all expected metrics are present
        expected_keys = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "jaccard",
            "hamming_loss",
        ]
        for key in expected_keys:
            self.assertIn(key, metrics)

    def test_compute_diagnostic_stats(self):
        """Test diagnostic statistics computation."""
        labels = np.array([[1, 0], [0, 1]])
        logits = np.array([[0.8, 0.2], [0.3, 0.7]])
        threshold = 0.5

        stats = compute_diagnostic_stats(labels, logits, threshold)

        self.assertEqual(stats["support"], 4)  # 2x2 array
        self.assertEqual(stats["threshold"], 0.5)
        self.assertEqual(stats["num_positive"], 2)
        self.assertGreaterEqual(stats["avg_probability"], 0)
        self.assertLessEqual(stats["avg_probability"], 1)


class TestMainMetricsFunction(unittest.TestCase):
    """Test the main compute_metrics function."""

    def test_compute_metrics_binary_classification(self):
        """Test compute_metrics for binary classification."""
        logits = [np.array([0.8]), np.array([0.2]), np.array([0.7])]
        labels = [np.array([1]), np.array([0]), np.array([1])]

        eval_pred = (logits, labels)
        metrics = compute_metrics(eval_pred, activated=False)

        # Should include basic binary metrics
        required_metrics = ["accuracy", "precision", "recall", "f1"]
        for metric in required_metrics:
            self.assertIn(metric, metrics)

    def test_compute_metrics_multilabel_classification(self):
        """Test compute_metrics for multilabel classification."""
        logits = [np.array([0.8, 0.2, 0.6]), np.array([0.1, 0.9, 0.3])]
        labels = [np.array([1, 0, 1]), np.array([0, 1, 0])]

        eval_pred = (logits, labels)
        metrics = compute_metrics(eval_pred, activated=False)

        # Should include multilabel-specific metrics
        multilabel_metrics = ["hamming_score", "jaccard_micro", "coverage_error"]
        for metric in multilabel_metrics:
            self.assertIn(metric, metrics)

    def test_compute_metrics_single_label_multiclass(self):
        """Test compute_metrics for single-label multiclass (one-hot)."""
        logits = [np.array([0.1, 0.8, 0.1]), np.array([0.7, 0.1, 0.2])]
        labels = [np.array([0, 1, 0]), np.array([1, 0, 0])]

        eval_pred = (logits, labels)
        metrics = compute_metrics(eval_pred, activated=False)

        # Should include match_accuracy for one-hot classification
        self.assertIn("match_accuracy", metrics)
        self.assertIn("accuracy", metrics)

    def test_compute_metrics_activated_vs_non_activated(self):
        """Test that activated flag works correctly."""
        # Test with pre-activated probabilities
        logits = [np.array([0.8, 0.2]), np.array([0.1, 0.9])]
        labels = [np.array([1, 0]), np.array([0, 1])]

        eval_pred = (logits, labels)

        # With activated=True, should include diagnostic stats
        metrics_activated = compute_metrics(eval_pred, activated=True)
        self.assertIn("support", metrics_activated)
        self.assertIn("threshold", metrics_activated)

        # With activated=False, should not include diagnostic stats
        metrics_not_activated = compute_metrics(eval_pred, activated=False)
        self.assertNotIn("support", metrics_not_activated)


class TestTopKMetrics(unittest.TestCase):
    """Test top-k ranking metrics."""

    def test_hit_rate_at_k_perfect(self):
        """Test Hit Rate at K with perfect ranking."""
        labels_list = [np.array([1, 0, 0, 0]), np.array([0, 0, 1, 0])]
        logits_list = [np.array([0.9, 0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.9, 0.3])]

        hr = hit_rate_at_k(labels_list, logits_list, k=1)
        self.assertEqual(hr, 1.0)  # Both top-1 predictions are correct

    def test_hit_rate_at_k_no_hits(self):
        """Test Hit Rate at K with no hits."""
        labels_list = [np.array([1, 0, 0, 0]), np.array([0, 0, 1, 0])]
        logits_list = [np.array([0.1, 0.9, 0.8, 0.7]), np.array([0.9, 0.8, 0.1, 0.7])]

        hr = hit_rate_at_k(labels_list, logits_list, k=1)
        self.assertEqual(hr, 0.0)  # No top-1 predictions are correct

    def test_hit_rate_at_k_bounds(self):
        """Test Hit Rate at K is within bounds."""
        labels_list = [np.array([1, 0, 0, 1]), np.array([0, 1, 1, 0])]
        logits_list = [np.array([0.8, 0.2, 0.1, 0.9]), np.array([0.1, 0.9, 0.8, 0.2])]

        hr = hit_rate_at_k(labels_list, logits_list, k=2)
        self.assertGreaterEqual(hr, 0.0)
        self.assertLessEqual(hr, 1.0)

    def test_mean_reciprocal_rank_at_k(self):
        """Test Mean Reciprocal Rank at K."""
        labels_list = [np.array([1, 0, 0, 0]), np.array([0, 0, 1, 0])]
        logits_list = [np.array([0.9, 0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.9, 0.3])]

        mrr = mean_reciprocal_rank_at_k(labels_list, logits_list, k=3)
        self.assertGreaterEqual(mrr, 0.0)
        self.assertLessEqual(mrr, 1.0)

    def test_precision_at_k(self):
        """Test Precision at K."""
        labels_list = [np.array([1, 0, 1, 0]), np.array([0, 1, 0, 1])]
        logits_list = [np.array([0.9, 0.1, 0.8, 0.2]), np.array([0.2, 0.9, 0.1, 0.8])]

        p_at_k = precision_at_k(labels_list, logits_list, k=2)
        self.assertGreaterEqual(p_at_k, 0.0)
        self.assertLessEqual(p_at_k, 1.0)

    def test_prepare_topk_data(self):
        """Test data preparation for top-k metrics."""
        logits = [np.array([[0.1, 0.9]]), np.array([[0.8, 0.2]])]
        labels = [np.array([[0, 1]]), np.array([[1, 0]])]

        logits_topk, labels_topk = prepare_topk_data(logits, labels)

        self.assertEqual(len(logits_topk), 2)
        self.assertEqual(len(labels_topk), 2)
        self.assertEqual(logits_topk[0].shape, (2,))
        self.assertEqual(labels_topk[0].shape, (2,))

    def test_compute_topk_metrics(self):
        """Test comprehensive top-k metrics computation."""
        logits = [np.array([0.1, 0.9, 0.3]), np.array([0.8, 0.2, 0.7])]
        labels = [np.array([0, 1, 0]), np.array([1, 0, 1])]

        eval_pred = (logits, labels)
        metrics = compute_topk_metrics(eval_pred, top_k=[1, 2])

        # Should include base metrics plus top-k metrics
        self.assertIn("accuracy", metrics)
        for k in [1, 2]:
            self.assertIn(f"HR@{k}", metrics)
            self.assertIn(f"MRR@{k}", metrics)
            self.assertIn(f"Precision@{k}", metrics)


class TestThresholdOptimization(unittest.TestCase):
    """Test threshold optimization functions."""

    def test_find_best_threshold_f1(self):
        """Test finding optimal threshold for F1 score."""
        logits = np.array([0.1, 0.9, 0.3, 0.8, 0.2])
        labels = np.array([0, 1, 0, 1, 0])

        best_threshold, best_score = find_best_threshold(logits, labels, metric="f1")

        self.assertGreaterEqual(best_threshold, 0.0)
        self.assertLessEqual(best_threshold, 1.0)
        self.assertGreaterEqual(best_score, 0.0)
        self.assertLessEqual(best_score, 1.0)
        self.assertIsInstance(best_threshold, float)
        self.assertIsInstance(best_score, float)

    def test_find_best_threshold_different_metrics(self):
        """Test finding optimal threshold for different metrics."""
        logits = np.array([0.1, 0.9, 0.3, 0.8])
        labels = np.array([0, 1, 0, 1])

        for metric in ["f1", "accuracy", "precision", "recall"]:
            threshold, score = find_best_threshold(logits, labels, metric=metric)
            self.assertGreaterEqual(threshold, 0.0)
            self.assertLessEqual(threshold, 1.0)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_find_best_threshold_perfect_separation(self):
        """Test threshold optimization with perfectly separable data."""
        logits = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])

        best_threshold, best_score = find_best_threshold(logits, labels, metric="f1")

        # Should achieve perfect F1 score
        self.assertEqual(best_score, 1.0)
        # Threshold should be between 0.2 and 0.8
        self.assertGreater(best_threshold, 0.2)
        self.assertLess(best_threshold, 0.8)

    def test_compute_best_metrics_binary(self):
        """Test computing metrics with optimized threshold for binary classification."""
        logits = [0.1, 0.9, 0.3, 0.8, 0.2]
        labels = [0, 1, 0, 1, 0]

        metrics = compute_best_metrics(logits, labels, multi=False)

        expected_keys = ["accuracy", "precision", "recall", "f1", "threshold"]
        for key in expected_keys:
            self.assertIn(key, metrics)

        self.assertIsNotNone(metrics["threshold"])

        # All metrics should be in valid range
        for key in ["accuracy", "precision", "recall", "f1"]:
            self.assertGreaterEqual(metrics[key], 0.0)
            self.assertLessEqual(metrics[key], 1.0)

    def test_compute_best_metrics_multiclass(self):
        """Test computing metrics for multiclass classification."""
        logits = [0, 1, 2, 1, 0]  # Class indices
        labels = [0, 1, 2, 1, 2]  # True class indices

        with patch("builtins.print"):  # Mock print to avoid output during testing
            metrics = compute_best_metrics(logits, labels, multi=True)

        expected_keys = ["accuracy", "precision", "recall", "f1", "threshold"]
        for key in expected_keys:
            self.assertIn(key, metrics)

        self.assertIsNone(metrics["threshold"])  # Should be None for multi-class

        # All metrics should be in valid range
        for key in ["accuracy", "precision", "recall", "f1"]:
            self.assertGreaterEqual(metrics[key], 0.0)
            self.assertLessEqual(metrics[key], 1.0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_single_sample_handling(self):
        """Test handling of single sample."""
        logits = [np.array([0.8])]
        labels = [np.array([1])]

        eval_pred = (logits, labels)
        metrics = compute_metrics(eval_pred)

        self.assertIn("accuracy", metrics)
        # Should handle single sample gracefully

    def test_all_zero_predictions(self):
        """Test handling when all predictions are zero."""
        labels = np.array([1, 1, 1])
        predictions = np.array([0, 0, 0])

        metrics = compute_basic_metrics(labels, predictions)

        # Should handle gracefully with zero_division=0
        self.assertEqual(metrics["precision"], 0)
        self.assertGreaterEqual(metrics["recall"], 0)
        self.assertLessEqual(metrics["recall"], 1)

    def test_all_zero_labels(self):
        """Test handling when all labels are zero."""
        labels = np.array([0, 0, 0])
        predictions = np.array([1, 1, 1])

        metrics = compute_basic_metrics(labels, predictions)

        # Should handle gracefully
        self.assertGreaterEqual(metrics["precision"], 0)
        self.assertLessEqual(metrics["precision"], 1)
        self.assertEqual(metrics["recall"], 0)

    def test_perfect_predictions_all_metrics(self):
        """Test that perfect predictions give expected results."""
        logits = [np.array([0.9, 0.1]), np.array([0.1, 0.9])]
        labels = [np.array([1, 0]), np.array([0, 1])]

        eval_pred = (logits, labels)
        metrics = compute_metrics(eval_pred, activated=True, threshold=0.5)

        # With perfect predictions and good threshold, should get high accuracy
        self.assertGreaterEqual(metrics["accuracy"], 0.9)

    def test_roc_auc_error_handling(self):
        """Test ROC AUC error handling for edge cases."""
        # Case where ROC AUC cannot be computed (all same class)
        labels = np.array([[1, 1, 1]])
        predictions = np.array([[1, 0, 1]])
        logits = np.array([[0.8, 0.2, 0.9]])

        # Should not raise an error, but skip ROC AUC
        metrics = compute_multilabel_metrics(labels, predictions, logits)

        # Should still compute other metrics
        self.assertIn("hamming_score", metrics)

    def test_empty_true_labels_precision_at_k(self):
        """Test precision at k when there are no true labels."""
        labels_list = [np.array([0, 0, 0, 0])]  # No true labels
        logits_list = [np.array([0.9, 0.8, 0.7, 0.6])]

        p_at_k = precision_at_k(labels_list, logits_list, k=2)

        # Should return 0.0 when no labels to evaluate
        self.assertEqual(p_at_k, 0.0)


class TestConsistencyAndValidation(unittest.TestCase):
    """Test consistency and validation of metrics."""

    def test_sigmoid_consistency(self):
        """Test sigmoid application consistency."""
        raw_logits = np.array([-1.0, 2.0, 0.5, -0.5])

        # Manual sigmoid calculation
        expected = 1 / (1 + np.exp(-raw_logits))
        result = apply_sigmoid(raw_logits)

        np.testing.assert_array_almost_equal(result, expected, decimal=10)

    def test_metric_bounds_validation(self):
        """Test that all metrics stay within expected bounds."""
        # Generate random test data
        np.random.seed(42)  # For reproducibility
        logits = [np.random.rand(3) for _ in range(10)]
        labels = [np.random.randint(0, 2, 3) for _ in range(10)]

        eval_pred = (logits, labels)
        metrics = compute_metrics(eval_pred, activated=True)

        # Test bounds for various metrics
        bounded_metrics = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "hamming_score",
            "jaccard_micro",
            "hamming_loss",
        ]

        for metric_name in bounded_metrics:
            if metric_name in metrics:
                self.assertGreaterEqual(
                    metrics[metric_name],
                    0,
                    f"{metric_name} should be >= 0, got {metrics[metric_name]}",
                )
                self.assertLessEqual(
                    metrics[metric_name],
                    1,
                    f"{metric_name} should be <= 1, got {metrics[metric_name]}",
                )

    def test_threshold_monotonicity(self):
        """Test that threshold affects predictions monotonically."""
        logits = np.array([0.2, 0.4, 0.6, 0.8])

        pred_low = get_predictions(logits, threshold=0.3)
        pred_mid = get_predictions(logits, threshold=0.5)
        pred_high = get_predictions(logits, threshold=0.7)

        # Higher threshold should result in fewer positive predictions
        self.assertGreaterEqual(pred_low.sum(), pred_mid.sum())
        self.assertGreaterEqual(pred_mid.sum(), pred_high.sum())


if __name__ == "__main__":
    # Create a test suite with all test classes
    test_classes = [
        TestUtilityFunctions,
        TestCustomMetrics,
        TestMetricComputation,
        TestMainMetricsFunction,
        TestTopKMetrics,
        TestThresholdOptimization,
        TestEdgeCases,
        TestConsistencyAndValidation,
    ]

    # Run tests with detailed output
    unittest.main(verbosity=2, exit=False)

    # Alternative: run specific test suites
    print("\n" + "=" * 60)
    print("RUNNING UNITTEST SUITE FOR METRICS MODULE")
    print("=" * 60)

    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=1)
        result = runner.run(suite)

        if result.wasSuccessful():
            print(f"✅ {test_class.__name__} - All tests passed!")
        else:
            print(
                f"❌ {test_class.__name__} - {len(result.failures)} failures, {len(result.errors)} errors"
            )
