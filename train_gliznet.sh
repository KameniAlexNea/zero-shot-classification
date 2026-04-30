#!/bin/bash
# GliZNet Training Script
# 
# This script trains a GliZNet model using the improved configuration system.
# Run with: bash train_gliznet.sh

# Generate timestamp for unique output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Use accelerate launch for DistributedDataParallel (DDP) — HuggingFace-native,
# auto-detects GPUs, and ensures torch.autocast propagates correctly per process.

nohup accelerate launch train_gliznet.py \
    \
    `# Model Configuration` \
    --model_name microsoft/deberta-v3-base \
    --model_class DebertaV2PreTrainedModel \
    --dropout_rate 0.1 \
    \
    `# Loss Configuration (SupCon + VICReg + BCE)` \
    --bce_loss_weight 0.5 \
    --supcon_loss_weight 0.5 \
    --label_repulsion_weight 0.05 \
    --supcon_margin 0.5 \
    --scoring_method bilinear \
    \
    `# Data Configuration` \
    --dataset_path alexneakameni/ZSHOT-HARDSET-v2 \
    --max_labels 20 \
    --shuffle_labels \
    --min_label_length 3 \
    --data_seed 42 \
    --max_extended_ds_size 20000 \
    --use_additional_datasets \
    \
    `# Tokenizer Configuration` \
    --use_fast_tokenizer \
    --model_max_length 512 \
    --lab_cls_token "[LAB]" \
    --max_tokens_per_span 16 \
    --min_text_tokens 5 \
    --min_label_tokens 1 \
    \
    `# Training Arguments` \
    --run_name "gliznet_training_${TIMESTAMP}" \
    --output_dir "results/deberta-v3-base_${TIMESTAMP}" \
    --num_train_epochs 10 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --learning_rate 4e-5 \
    --warmup_steps 0.05 \
    --weight_decay 1e-3 \
    --lr_scheduler_type cosine \
    --max_grad_norm 2.0 \
    \
    `# Evaluation & Checkpointing` \
    --eval_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 4 \
    --load_best_model_at_end \
    --metric_for_best_model loss \
    --early_stopping_patience 3 \
    --eval_on_start \
    --eval_do_concat_batches False \
    \
    `# Performance Optimization` \
    --dataloader_pin_memory \
    --dataloader_num_workers 8 \
    --dataloader_prefetch_factor 2 \
    --eval_use_gather_object \
    --ddp_find_unused_parameters False \
    --torch_compile \
    --bf16 \
    \
    `# Logging & Monitoring` \
    --logging_steps 100 \
    --report_to wandb \
    --remove_unused_columns False \
    \
    &> nohup.out &

echo "Training started in background (PID: $!)"
echo "Monitor progress with: tail -f nohup.out"