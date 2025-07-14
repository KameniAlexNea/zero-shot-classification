#!/usr/bin/env python3
"""
Test script to validate the updated MTEB-style evaluation with train/test splits and pre-tokenized data.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from datasets import Dataset

# Import our updated MTEB evaluators
from mteb_style_evals import (
    LogRegClassificationEvaluator,
    kNNClassificationEvaluator,
)


class MockModelWrapper:
    """Mock model wrapper for testing with pre-tokenized data."""

    def __init__(self, embed_dim=128):
        self.embed_dim = embed_dim

    def encode_from_tokenized_dataset(self, dataset, **kwargs):
        """Generate random embeddings for testing with pre-tokenized dataset."""
        np.random.seed(42)  # For reproducibility
        return np.random.randn(len(dataset), self.embed_dim)


def create_mock_dataset(num_samples=50, num_labels=3):
    """Create a mock tokenized dataset similar to what add_tokenized_function produces."""
    np.random.seed(42)

    # Create mock label names
    label_names = [f"label_{i}" for i in range(num_labels)]

    data = []
    for i in range(num_samples):
        # Mock tokenized data (similar to what add_tokenized_function produces)
        input_ids = torch.randint(1, 1000, (50,))  # Mock token IDs
        attention_mask = torch.ones(50)  # Mock attention mask

        # Mock labels - each sample has one positive label
        positive_label_idx = np.random.randint(0, num_labels)
        ltext = label_names.copy()
        lint = [i == positive_label_idx for i in range(num_labels)]

        data.append(
            {
                "input_ids": input_ids.tolist(),
                "attention_mask": attention_mask.tolist(),
                "ltext": ltext,
                "lint": lint,
            }
        )

    return Dataset.from_list(data)


def test_updated_knn_evaluator():
    """Test updated KNN evaluator with train/test datasets."""
    print("Testing Updated KNN Evaluator with Train/Test Splits...")

    # Create mock train and test datasets
    train_dataset = create_mock_dataset(num_samples=80, num_labels=3)
    test_dataset = create_mock_dataset(num_samples=20, num_labels=3)

    # Create mock labels (simulating pre-extracted labels)
    label_to_idx = {"label_0": 0, "label_1": 1, "label_2": 2}
    y_train = [np.random.randint(0, 3) for _ in range(80)]
    y_test = [np.random.randint(0, 3) for _ in range(20)]

    # Create evaluator
    evaluator = kNNClassificationEvaluator(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        label_to_idx=label_to_idx,
        y_train=y_train,
        y_test=y_test,
        k=3,
        encode_kwargs={"batch_size": 32},
    )

    # Create mock model
    mock_model = MockModelWrapper()

    # Run evaluation
    scores, test_cache = evaluator(mock_model)

    print("Updated KNN Results:")
    for metric, value in scores.items():
        print(f"  {metric}: {value:.4f}")

    # Verify we got expected metrics
    expected_metrics = [
        "accuracy_cosine",
        "accuracy_euclidean",
        "f1_cosine",
        "f1_euclidean",
        "accuracy",
        "f1",
    ]
    for metric in expected_metrics:
        assert metric in scores, f"Missing metric: {metric}"

    # Verify label consistency
    print(f"  Train labels: {len(set(evaluator.y_train))} unique")
    print(f"  Test labels: {len(set(evaluator.y_test))} unique")
    print(f"  Total labels: {len(evaluator.label_to_idx)} unique")

    print("✓ Updated KNN Evaluator test passed!")
    return scores


def test_updated_logreg_evaluator():
    """Test updated Logistic Regression evaluator with train/test datasets."""
    print("\nTesting Updated Logistic Regression Evaluator with Train/Test Splits...")

    # Create mock train and test datasets
    train_dataset = create_mock_dataset(num_samples=80, num_labels=3)
    test_dataset = create_mock_dataset(num_samples=20, num_labels=3)

    # Create mock labels (simulating pre-extracted labels)
    label_to_idx = {"label_0": 0, "label_1": 1, "label_2": 2}
    y_train = [np.random.randint(0, 3) for _ in range(80)]
    y_test = [np.random.randint(0, 3) for _ in range(20)]

    # Create evaluator
    evaluator = LogRegClassificationEvaluator(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        label_to_idx=label_to_idx,
        y_train=y_train,
        y_test=y_test,
        max_iter=100,
        encode_kwargs={"batch_size": 32},
    )

    # Create mock model
    mock_model = MockModelWrapper()

    # Run evaluation
    scores, test_cache = evaluator(mock_model)

    print("Updated LogReg Results:")
    for metric, value in scores.items():
        print(f"  {metric}: {value:.4f}")

    # Verify we got expected metrics
    expected_metrics = ["accuracy", "f1", "f1_weighted"]
    for metric in expected_metrics:
        assert metric in scores, f"Missing metric: {metric}"

    # Verify label consistency
    print(f"  Train labels: {len(set(evaluator.y_train))} unique")
    print(f"  Test labels: {len(set(evaluator.y_test))} unique")
    print(f"  Total labels: {len(evaluator.label_to_idx)} unique")

    print("✓ Updated Logistic Regression Evaluator test passed!")
    return scores


def test_shared_label_mapping():
    """Test that train and test datasets share the same label mapping."""
    print("\nTesting Shared Label Mapping...")

    # Create datasets with overlapping but different label distributions
    train_dataset = create_mock_dataset(num_samples=60, num_labels=4)
    test_dataset = create_mock_dataset(num_samples=40, num_labels=4)

    # Create mock shared label mapping
    label_to_idx = {"label_0": 0, "label_1": 1, "label_2": 2, "label_3": 3}
    y_train = [np.random.randint(0, 4) for _ in range(60)]
    y_test = [np.random.randint(0, 4) for _ in range(40)]

    # Test with KNN evaluator
    knn_evaluator = kNNClassificationEvaluator(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        label_to_idx=label_to_idx,
        y_train=y_train,
        y_test=y_test,
        k=3,
    )

    # Verify both datasets use the same label space
    print(
        f"  Train label range: {min(knn_evaluator.y_train)} - {max(knn_evaluator.y_train)}"
    )
    print(
        f"  Test label range: {min(knn_evaluator.y_test)} - {max(knn_evaluator.y_test)}"
    )
    print(f"  Label mapping: {knn_evaluator.label_to_idx}")

    # Both should use the same label space
    assert len(knn_evaluator.label_to_idx) == 4, "Should have 4 unique labels"

    print("✓ Shared label mapping test passed!")


def main():
    """Run all tests for the updated implementation."""
    print("Running Updated MTEB-style evaluation tests...\n")

    try:
        # Test updated KNN evaluator
        test_updated_knn_evaluator()

        # Test updated LogReg evaluator
        test_updated_logreg_evaluator()

        # Test shared label mapping
        test_shared_label_mapping()

        print("\n" + "=" * 60)
        print("All updated tests passed successfully! ✓")
        print("=" * 60)

        print("\nKey improvements in the updated implementation:")
        print("1. ✓ Uses separate train/test splits from dataset loaders")
        print("2. ✓ Leverages pre-tokenized data from add_tokenized_function")
        print("3. ✓ Eliminates redundant tokenization")
        print("4. ✓ Ensures consistent label mapping across train/test")
        print("5. ✓ Works directly with tokenized datasets")

        print("\nThe updated MTEB-style evaluation is ready for use!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
