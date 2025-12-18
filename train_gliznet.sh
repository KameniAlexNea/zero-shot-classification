#!/bin/bash
# GliZNet Training Script
# 
# This script trains a GliZNet model using the improved configuration system.
# Run with: bash train_gliznet.sh

nohup uv run train_gliznet.py \
    \
    `# Model Configuration` \
    --model_name microsoft/deberta-v3-small \
    --model_class DebertaV2PreTrainedModel \
    --projected_dim 512 \
    --similarity_metric dot_learning \
    --dropout_rate 0.1 \
    \
    `# Loss Configuration` \
    --scale_loss 10.0 \
    --margin 0.1 \
    --temperature 1.0 \
    --contrastive_loss_weight 1.0 \
    --use_separator_pooling \
    \
    `# Data Configuration` \
    --dataset_path alexneakameni/ZSHOT-HARDSET \
    --dataset_name triplet \
    --max_labels 15 \
    --shuffle_labels \
    --min_label_length 2 \
    --data_seed 42 \
    \
    `# Tokenizer Configuration` \
    --use_fast_tokenizer \
    --model_max_length 512 \
    --cls_separator_token "[LAB]" \
    \
    `# Training Arguments` \
    --run_name "gliznet_training" \
    --output_dir "results/deberta-v3-small-sep-pooling" \
    --num_train_epochs 4 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 8 \
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
    --save_total_limit 2 \
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
    &> nohup1.out &

echo "Training started in background (PID: $!)"
echo "Monitor progress with: tail -f nohup1.out"