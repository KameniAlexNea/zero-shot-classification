"""
Essential unittest tests for metrics module.
Focused on the most critical functions with clear, simple test cases.
"""

import os
import sys
import unittest

import numpy as np

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.metrics import (
    apply_sigmoid,
    compute_metrics,
    find_best_threshold,
    flatten_nested_lists,
    hamming_score,
    is_multilabel,
    is_single_label_multiclass,
)


class TestBasicFunctions(unittest.TestCase):
    """Test basic utility functions."""

    def test_flatten_nested_lists(self):
        """Test list flattening functionality."""
        # Single level nesting
        data = [[1, 2], [3, 4]]
        result = flatten_nested_lists(data)
        self.assertEqual(result, [1, 2, 3, 4])

        # Already flat
        data = [1, 2, 3, 4]
        result = flatten_nested_lists(data)
        self.assertEqual(result, [1, 2, 3, 4])

        # Empty list
        data = []
        result = flatten_nested_lists(data)
        self.assertEqual(result, [])

    def test_apply_sigmoid(self):
        """Test sigmoid activation function."""
        # Test known values
        logits = np.array([0, 1, -1])
        result = apply_sigmoid(logits)

        self.assertAlmostEqual(result[0], 0.5, places=6)  # sigmoid(0) = 0.5
        self.assertGreater(result[1], 0.5)  # sigmoid(1) > 0.5
        self.assertLess(result[2], 0.5)  # sigmoid(-1) < 0.5

        # Test monotonicity
        self.assertLess(result[2], result[0])
        self.assertLess(result[0], result[1])

    def test_is_multilabel(self):
        """Test multilabel detection."""
        # True multilabel (2D with multiple columns)
        labels = np.array([[1, 0, 1], [0, 1, 1]])
        self.assertTrue(is_multilabel(labels))

        # Binary (1D)
        labels = np.array([1, 0, 1, 0])
        self.assertFalse(is_multilabel(labels))

        # Single column 2D
        labels = np.array([[1], [0], [1]])
        self.assertFalse(is_multilabel(labels))

    def test_is_single_label_multiclass(self):
        """Test one-hot detection."""
        # One-hot encoded (exactly one 1 per row)
        labels = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertTrue(is_single_label_multiclass(labels))

        # True multilabel (multiple 1s per row)
        labels = np.array([[1, 1, 0], [0, 1, 1]])
        self.assertFalse(is_single_label_multiclass(labels))


class TestCustomMetrics(unittest.TestCase):
    """Test custom metric implementations."""

    def test_hamming_score_perfect(self):
        """Test Hamming score with perfect predictions."""
        y_true = np.array([[1, 0, 1], [0, 1, 0]])
        y_pred = np.array([[1, 0, 1], [0, 1, 0]])
        score = hamming_score(y_true, y_pred)
        self.assertEqual(score, 1.0)

    def test_hamming_score_no_match(self):
        """Test Hamming score with no matches."""
        y_true = np.array([[1, 0, 1], [0, 1, 0]])
        y_pred = np.array([[0, 1, 0], [1, 0, 1]])
        score = hamming_score(y_true, y_pred)
        self.assertEqual(score, 0.0)

    def test_hamming_score_all_zeros(self):
        """Test Hamming score with all zeros."""
        y_true = np.array([[0, 0, 0], [0, 0, 0]])
        y_pred = np.array([[0, 0, 0], [0, 0, 0]])
        score = hamming_score(y_true, y_pred)
        self.assertEqual(score, 1.0)


class TestComputeMetrics(unittest.TestCase):
    """Test the main compute_metrics function."""

    def test_binary_classification(self):
        """Test binary classification workflow."""
        # Simple binary case
        logits = [np.array([0.8]), np.array([0.2]), np.array([0.7]), np.array([0.1])]
        labels = [np.array([1]), np.array([0]), np.array([1]), np.array([0])]

        eval_pred = (logits, labels)
        metrics = compute_metrics(eval_pred, activated=False)

        # Should have basic metrics
        required_metrics = ["accuracy", "precision", "recall", "f1"]
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertGreaterEqual(metrics[metric], 0.0)
            self.assertLessEqual(metrics[metric], 1.0)

    def test_multilabel_classification(self):
        """Test multilabel classification workflow."""
        # Multilabel case
        logits = [
            np.array([0.8, 0.2, 0.6]),
            np.array([0.1, 0.9, 0.3]),
            np.array([0.7, 0.1, 0.8]),
        ]
        labels = [np.array([1, 0, 1]), np.array([0, 1, 0]), np.array([1, 0, 1])]

        eval_pred = (logits, labels)
        metrics = compute_metrics(eval_pred, activated=False)

        # Should have multilabel-specific metrics
        multilabel_metrics = ["jaccard", "hamming_loss"]
        for metric in multilabel_metrics:
            self.assertIn(metric, metrics)

    def test_single_label_multiclass(self):
        """Test one-hot encoded multiclass."""
        # One-hot encoded
        logits = [
            np.array([0.1, 0.8, 0.1]),
            np.array([0.7, 0.2, 0.1]),
            np.array([0.2, 0.1, 0.7]),
        ]
        labels = [np.array([0, 1, 0]), np.array([1, 0, 0]), np.array([0, 0, 1])]

        eval_pred = (logits, labels)
        metrics = compute_metrics(eval_pred, activated=False)

        # Should have match_accuracy
        self.assertIn("accuracy", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)

    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        # Perfect binary predictions
        logits = [np.array([0.9]), np.array([0.1]), np.array([0.8]), np.array([0.2])]
        labels = [np.array([1]), np.array([0]), np.array([1]), np.array([0])]

        eval_pred = (logits, labels)
        metrics = compute_metrics(eval_pred, activated=True, threshold=0.5)

        # Should get high accuracy with good predictions
        self.assertGreaterEqual(metrics["accuracy"], 0.75)


class TestThresholdOptimization(unittest.TestCase):
    """Test threshold optimization functions."""

    def test_find_best_threshold_basic(self):
        """Test basic threshold optimization."""
        # Separable data
        logits = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])

        best_threshold, best_score = find_best_threshold(logits, labels, metric="f1")

        # Should find a good threshold and score
        self.assertGreater(best_threshold, 0.2)
        self.assertLess(best_threshold, 0.8)
        self.assertGreater(best_score, 0.8)  # Should be high for separable data

    def test_find_best_threshold_perfect(self):
        """Test threshold optimization with perfect separation."""
        # Perfectly separable
        logits = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])

        best_threshold, best_score = find_best_threshold(
            logits, labels, metric="accuracy"
        )

        # Should achieve perfect accuracy
        self.assertEqual(best_score, 1.0)

    def test_find_best_threshold_different_metrics(self):
        """Test with different optimization metrics."""
        logits = np.array([0.3, 0.7, 0.4, 0.6])
        labels = np.array([0, 1, 0, 1])

        for metric in ["f1", "accuracy", "precision", "recall"]:
            threshold, score = find_best_threshold(logits, labels, metric=metric)
            self.assertGreaterEqual(threshold, 0.0)
            self.assertLessEqual(threshold, 1.0)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_single_sample(self):
        """Test with single sample."""
        logits = [np.array([0.8])]
        labels = [np.array([1])]

        eval_pred = (logits, labels)
        metrics = compute_metrics(eval_pred)

        # Should handle single sample
        self.assertIn("accuracy", metrics)

    def test_all_same_predictions(self):
        """Test when all predictions are the same."""
        logits = [np.array([0.9]), np.array([0.9]), np.array([0.9])]
        labels = [np.array([1]), np.array([0]), np.array([1])]

        eval_pred = (logits, labels)
        metrics = compute_metrics(eval_pred, activated=True, threshold=0.5)

        # Should handle gracefully
        self.assertIn("accuracy", metrics)

    def test_all_zero_labels(self):
        """Test with all zero labels."""
        logits = [np.array([0.3]), np.array([0.7]), np.array([0.4])]
        labels = [np.array([0]), np.array([0]), np.array([0])]

        eval_pred = (logits, labels)
        metrics = compute_metrics(eval_pred)

        # Should handle all zero labels
        self.assertIn("accuracy", metrics)


def run_simple_validation():
    """Run a simple validation check."""
    print("\n" + "=" * 50)
    print("SIMPLE VALIDATION CHECK")
    print("=" * 50)

    try:
        # Test basic functionality
        print("Testing basic functions...")

        # Test sigmoid
        result = apply_sigmoid(np.array([0]))
        assert abs(result[0] - 0.5) < 1e-6
        print("✅ Sigmoid function works")

        # Test multilabel detection
        labels = np.array([[1, 0, 1], [0, 1, 1]])
        assert is_multilabel(labels)
        print("✅ Multilabel detection works")

        # Test compute_metrics
        logits = [np.array([0.8]), np.array([0.2])]
        labels = [np.array([1]), np.array([0])]
        eval_pred = (logits, labels)
        metrics = compute_metrics(eval_pred)
        assert "accuracy" in metrics
        print("✅ Compute metrics works")

        print("\n🎉 All basic validations passed!")
        return True

    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        return False


if __name__ == "__main__":
    print("ESSENTIAL METRICS UNITTEST SUITE")
    print("=" * 50)

    # Run simple validation first
    if not run_simple_validation():
        sys.exit(1)

    # Run full unittest suite
    print("\nRunning full unittest suite...")
    unittest.main(verbosity=2)
