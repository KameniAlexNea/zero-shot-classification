#!/usr/bin/env python3
"""
Quick test script to verify MTEB-style evaluation works correctly.
This script tests the basic functionality without requiring a full model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# Import our MTEB evaluators
from mteb_style_evals import kNNClassificationEvaluator, LogRegClassificationEvaluator

class MockModelWrapper:
    """Mock model wrapper for testing."""
    
    def __init__(self, embed_dim=128):
        self.embed_dim = embed_dim
    
    def encode(self, sentences, **kwargs):
        """Generate random embeddings for testing."""
        np.random.seed(42)  # For reproducibility
        return np.random.randn(len(sentences), self.embed_dim)

def test_knn_evaluator():
    """Test KNN evaluator."""
    print("Testing KNN Evaluator...")
    
    # Generate mock data
    sentences_train = [f"Training sentence {i}" for i in range(100)]
    sentences_test = [f"Test sentence {i}" for i in range(20)]
    
    # Generate corresponding labels
    y_train = np.random.randint(0, 3, 100).tolist()
    y_test = np.random.randint(0, 3, 20).tolist()
    
    # Create evaluator
    evaluator = kNNClassificationEvaluator(
        sentences_train=sentences_train,
        y_train=y_train,
        sentences_test=sentences_test,
        y_test=y_test,
        k=3,
        encode_kwargs={"batch_size": 32}
    )
    
    # Create mock model
    mock_model = MockModelWrapper()
    
    # Run evaluation
    scores, test_cache = evaluator(mock_model)
    
    print("KNN Results:")
    for metric, value in scores.items():
        print(f"  {metric}: {value:.4f}")
    
    # Verify we got expected metrics
    expected_metrics = ["accuracy_cosine", "accuracy_euclidean", "f1_cosine", "f1_euclidean", "accuracy", "f1"]
    for metric in expected_metrics:
        assert metric in scores, f"Missing metric: {metric}"
    
    print("✓ KNN Evaluator test passed!")
    return scores

def test_logreg_evaluator():
    """Test Logistic Regression evaluator."""
    print("\nTesting Logistic Regression Evaluator...")
    
    # Generate mock data
    sentences_train = [f"Training sentence {i}" for i in range(100)]
    sentences_test = [f"Test sentence {i}" for i in range(20)]
    
    # Generate corresponding labels
    y_train = np.random.randint(0, 3, 100).tolist()
    y_test = np.random.randint(0, 3, 20).tolist()
    
    # Create evaluator
    evaluator = LogRegClassificationEvaluator(
        sentences_train=sentences_train,
        y_train=y_train,
        sentences_test=sentences_test,
        y_test=y_test,
        max_iter=100,
        encode_kwargs={"batch_size": 32}
    )
    
    # Create mock model
    mock_model = MockModelWrapper()
    
    # Run evaluation
    scores, test_cache = evaluator(mock_model)
    
    print("LogReg Results:")
    for metric, value in scores.items():
        print(f"  {metric}: {value:.4f}")
    
    # Verify we got expected metrics
    expected_metrics = ["accuracy", "f1", "f1_weighted"]
    for metric in expected_metrics:
        assert metric in scores, f"Missing metric: {metric}"
    
    print("✓ Logistic Regression Evaluator test passed!")
    return scores

def test_binary_classification():
    """Test binary classification metrics."""
    print("\nTesting Binary Classification...")
    
    # Generate binary classification data
    sentences_train = [f"Training sentence {i}" for i in range(100)]
    sentences_test = [f"Test sentence {i}" for i in range(20)]
    
    # Binary labels
    y_train = np.random.randint(0, 2, 100).tolist()
    y_test = np.random.randint(0, 2, 20).tolist()
    
    # Test KNN with binary classification
    evaluator = kNNClassificationEvaluator(
        sentences_train=sentences_train,
        y_train=y_train,
        sentences_test=sentences_test,
        y_test=y_test,
        k=3
    )
    
    mock_model = MockModelWrapper()
    scores, _ = evaluator(mock_model)
    
    # Should have AP metrics for binary classification
    assert "ap" in scores, "Missing AP metric for binary classification"
    assert "ap_cosine" in scores, "Missing AP cosine metric for binary classification"
    
    print("Binary classification metrics:")
    for metric, value in scores.items():
        if "ap" in metric:
            print(f"  {metric}: {value:.4f}")
    
    print("✓ Binary classification test passed!")

def main():
    """Run all tests."""
    print("Running MTEB-style evaluation tests...\n")
    
    try:
        # Test KNN evaluator
        test_knn_evaluator()
        
        # Test LogReg evaluator  
        test_logreg_evaluator()
        
        # Test binary classification
        test_binary_classification()
        
        print("\n" + "="*50)
        print("All tests passed successfully! ✓")
        print("="*50)
        
        print("\nThe MTEB-style evaluation implementation is working correctly.")
        print("You can now use it with your trained GliZNet model.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
