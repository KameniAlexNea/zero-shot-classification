#!/bin/bash
# GliZNet Training Script
# 
# This script trains a GliZNet model using the improved configuration system.
# Run with: bash train_gliznet.sh

# Generate timestamp for unique output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

nohup uv run train_gliznet.py \
    \
    `# Model Configuration` \
    --model_name alexneakameni/checkpoint-178 \
    --similarity_metric cosine \
    --dropout_rate 0.1 \
    \
    `# Loss Configuration (SupCon + Label Repulsion + BCE)` \
    --bce_loss_weight 1.0 \
    --supcon_loss_weight 1.0 \
    --label_repulsion_weight 0.1 \
    --logit_scale_init 2.0 \
    --learn_temperature \
    --repulsion_threshold 0.3 \
    \
    `# Data Configuration` \
    --dataset_path alexneakameni/synthetic-classification-dataset \
    --max_labels 15 \
    --shuffle_labels \
    --min_label_length 2 \
    --data_seed 42 \
    --max_extended_ds_size 5000 \
    \
    `# Tokenizer Configuration` \
    --use_fast_tokenizer \
    --model_max_length 512 \
    --lab_cls_token "[LAB]" \
    \
    `# Training Arguments` \
    --run_name "gliznet_training_${TIMESTAMP}" \
    --output_dir "results/modern-bert-base_${TIMESTAMP}" \
    --num_train_epochs 4 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.01 \
    --weight_decay 1e-3 \
    --lr_scheduler_type cosine \
    \
    `# Evaluation & Checkpointing` \
    --eval_strategy steps \
    --eval_steps 0.25 \
    --save_strategy steps \
    --save_steps 0.25 \
    --save_total_limit 4 \
    --load_best_model_at_end \
    --metric_for_best_model eval_loss \
    --early_stopping_patience 3 \
    --eval_on_start \
    --eval_do_concat_batches False \
    \
    `# Performance Optimization` \
    --dataloader_pin_memory \
    --dataloader_num_workers 16 \
    --dataloader_prefetch_factor 2 \
    --dataloader_drop_last \
    --fp16 \
    \
    `# Logging & Monitoring` \
    --logging_steps 100 \
    --report_to wandb \
    --remove_unused_columns False \
    \
    &> nohup.out &

echo "Training started in background (PID: $!)"
echo "Monitor progress with: tail -f nohup.out"