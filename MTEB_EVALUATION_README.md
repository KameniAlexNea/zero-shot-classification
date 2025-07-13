# MTEB-Style Evaluation vs Original GliZNet Evaluation

This document explains the key differences between the original GliZNet evaluation approach and the new MTEB-style evaluation.

## Original GliZNet Evaluation (`evaluate_agnews.py`)

### Approach:
- **End-to-end**: Uses the full GliZNet model with its specialized tokenization and label masking
- **Model-specific**: Leverages the model's built-in `predict()` method that computes similarities between CLS tokens and label embeddings
- **Label-aware**: Maintains the original zero-shot classification paradigm where labels are tokenized alongside text

### Process:
1. Tokenizes text + labels together using `GliZNETTokenizer`
2. Creates label masks (`lmask`) to identify label tokens
3. Uses model's forward pass with attention mechanisms
4. Computes similarity scores between text and label representations
5. Applies sigmoid/softmax activation for final predictions

### Metrics:
- Top-k accuracy, precision, recall, F1
- Threshold-based binary classification metrics
- Custom metrics designed for zero-shot classification

## MTEB-Style Evaluation (`mteb_style_evals.py`)

### Approach:
- **Embedding-based**: Uses only the model's encoding capability (`.encode()` method)
- **Model-agnostic**: Treats the model as a general sentence encoder
- **Traditional classification**: Converts zero-shot task to supervised classification

### Process:
1. Encodes text sentences only (no labels) using model's `.encode()` method
2. Extracts CLS token embeddings as sentence representations
3. Prepares train/test splits from the dataset
4. Trains traditional classifiers (KNN or Logistic Regression) on embeddings
5. Evaluates using standard classification metrics

### Classifiers Available:
- **KNN**: k-Nearest Neighbors with cosine/euclidean distance
- **LogReg**: Logistic Regression with L2 regularization

### Metrics:
- Accuracy, F1-score (macro/weighted)
- Average Precision (for binary tasks)
- Separate metrics for different distance functions (KNN only)

## Key Differences

| Aspect | Original GliZNet | MTEB-Style |
|--------|------------------|------------|
| **Paradigm** | Zero-shot classification | Supervised classification |
| **Labels** | Encoded with text | Used only for train/test split |
| **Model Usage** | Full forward pass | Encoding only |
| **Training** | None (zero-shot) | Classifier training required |
| **Similarity** | Learned similarity function | Distance-based (cosine/euclidean) |
| **Evaluation** | Domain-specific metrics | Standard ML metrics |
| **Interpretability** | Label-text relationships | Embedding space structure |

## When to Use Which

### Use Original GliZNet Evaluation when:
- Evaluating the model's zero-shot capabilities
- Testing label-text similarity learning
- Comparing with other zero-shot classification methods
- Analyzing the model's attention mechanisms

### Use MTEB-Style Evaluation when:
- Comparing with general sentence encoders
- Evaluating embedding quality
- Testing on MTEB benchmark tasks
- Analyzing representation learning capabilities

## Usage Examples

### Original Evaluation:
```bash
python evaluate_agnews.py \
    --model_path "results/best_model/model" \
    --data "events_biotech" \
    --activation "sigmoid" \
    --results_dir "results/evaluation_original"
```

### MTEB-Style Evaluation:
```bash
python mteb_style_evals.py \
    --model_path "results/best_model/model" \
    --data "events_biotech" \
    --classifier_type "knn" \
    --k 5 \
    --results_dir "results/mteb_evaluation"
```

## Output Comparison

### Original GliZNet Output:
- Per-label similarity scores
- Zero-shot classification probabilities
- Attention-weighted label representations

### MTEB-Style Output:
- Dense embeddings for each sentence
- Traditional classification predictions
- Distance-based similarity scores

Both evaluation methods provide valuable insights into different aspects of the model's performance and can be used complementarily to get a comprehensive understanding of the model's capabilities.
