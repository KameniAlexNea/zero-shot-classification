#!/bin/bash

# Example usage of MTEB-style evaluation script

# Set environment variables
export WANDB_PROJECT="zero-shot-classification"
export WANDB_WATCH="none"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

# Run MTEB-style evaluation with KNN classifier
echo "Running MTEB-style evaluation with KNN classifier..."
python mteb_style_evals.py \
    --model_path "results/best_model/model" \
    --data "events_biotech" \
    --classifier_type "knn" \
    --k 5 \
    --batch_size 32 \
    --device_pos 0 \
    --results_dir "results/mteb_evaluation_knn" \
    --model_class "DebertaV2PreTrainedModel" \
    --use_fast_tokenizer

echo "KNN evaluation completed!"

# Run MTEB-style evaluation with Logistic Regression classifier
echo "Running MTEB-style evaluation with Logistic Regression classifier..."
python mteb_style_evals.py \
    --model_path "results/best_model/model" \
    --data "events_biotech" \
    --classifier_type "logreg" \
    --max_iter 1000 \
    --batch_size 32 \
    --device_pos 0 \
    --results_dir "results/mteb_evaluation_logreg" \
    --model_class "DebertaV2PreTrainedModel" \
    --use_fast_tokenizer

echo "Logistic Regression evaluation completed!"

# Optional: limit samples for quick testing
echo "Running quick test with limited samples..."
python mteb_style_evals.py \
    --model_path "results/best_model/model" \
    --data "events_biotech" \
    --classifier_type "knn" \
    --k 3 \
    --batch_size 16 \
    --limit 100 \
    --device_pos 0 \
    --results_dir "results/mteb_evaluation_quick_test" \
    --model_class "DebertaV2PreTrainedModel" \
    --use_fast_tokenizer

echo "Quick test completed!"

echo "All MTEB-style evaluations completed! Check the results directories for detailed metrics."
