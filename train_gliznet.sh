#!/bin/bash
# GliZNet Training Script
# 
# This script trains a GliZNet model using the improved configuration system.
# Run with: bash train_gliznet.sh

# Generate timestamp for unique output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

MODEL_PATH="microsoft/deberta-v3-base"

# Use accelerate launch for DistributedDataParallel (DDP) — HuggingFace-native,
# auto-detects GPUs, and ensures torch.autocast propagates correctly per process.

nohup accelerate launch train_gliznet.py \
    \
    `# Model Configuration` \
    --model_name "${MODEL_PATH}" \
    --model_class DebertaV2PreTrainedModel \
    --dropout_rate 0.1 \
    --enrich_labels \
    --save_only_model \
    \
    `# Loss Configuration (SupCon + VICReg + BCE) — matching best run config` \
    --focal_loss_weight 0.8 \
    --focal_gamma 1.85 \
    --supcon_loss_weight 1.0 \
    --label_repulsion_weight 0.1 \
    --supcon_margin 0.1 \
    --scoring_method bilinear \
    \
    `# Data Configuration` \
    --dataset_path alexneakameni/ZSHOT-HARDSET-v2 \
    --max_labels 20 \
    --shuffle_labels \
    --min_label_length 3 \
    --data_seed 42 \
    --max_extended_ds_size 5000 \
    --use_additional_datasets \
    --text_augmentation True \
    --augmentation_config gliznet/config/augmentation_config.yaml \
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
    --num_train_epochs 4 \
    --per_device_train_batch_size 48 \
    --per_device_eval_batch_size 48 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --warmup_steps 0.05 \
    --weight_decay 1e-3 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    \
    `# Evaluation & Checkpointing` \
    --eval_strategy steps \
    --save_strategy steps \
    --eval_steps 4000 \
    --save_steps 4000 \
    --save_total_limit 5 \
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
    `# --torch_compile` \
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