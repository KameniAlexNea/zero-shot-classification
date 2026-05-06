#!/bin/bash
# GliZNet Training Script
# 
# This script trains a GliZNet model using the improved configuration system.
# Run with: bash train_gliznet.sh

# Generate timestamp for unique output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MODEL_PATH="results/pretrained_alex_20260506_063055/checkpoint-1681"

# Use accelerate launch for DistributedDataParallel (DDP) — HuggingFace-native,
# auto-detects GPUs, and ensures torch.autocast propagates correctly per process.

nohup accelerate launch train_gliznet.py \
    \
    `# Model Configuration` \
    --model_name "${MODEL_PATH}" \
    --model_class DebertaV2PreTrainedModel \
    --dropout_rate 0.1 \
    \
    `# Loss Configuration (SupCon + VICReg + BCE) — matching best run config` \
    --focal_loss_weight 0.3 \
    --focal_gamma 1.8 \
    --supcon_loss_weight 1.0 \
    --label_repulsion_weight 0.1 \
    --supcon_margin 0.1 \
    --scoring_method bilinear \
    \
    `# Data Configuration` \
    --dataset_path alexneakameni/ZSHOT-HARDSET-Polarity \
    --max_labels 40 \
    --shuffle_labels \
    --min_label_length 3 \
    --data_seed 42 \
    --text_augmentation True \
    --max_extended_ds_size 50000 \
    --use_additional_datasets \
    --augmentation_config config/augmentation_config.yaml \
    `# Tokenizer Configuration` \
    --use_fast_tokenizer \
    --model_max_length 512 \
    --lab_cls_token "[LAB]" \
    --max_tokens_per_span 16 \
    --min_text_tokens 5 \
    --min_label_tokens 1 \
    --enrich_labels \
    \
    `# Training Arguments` \
    --run_name "gliznet_pretrain_alex_${TIMESTAMP}" \
    --output_dir "results/pretrained_alex_${TIMESTAMP}" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --learning_rate 4e-5 \
    --warmup_steps 0.05 \
    --weight_decay 1e-3 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    \
    `# Evaluation & Checkpointing` \
    --eval_strategy steps \
    --save_strategy steps \
    --eval_steps 500 \
    --save_steps 500 \
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
    `# --torch_compile` \
    --bf16 \
    \
    `# Logging & Monitoring` \
    --logging_steps 100 \
    --report_to wandb \
    --remove_unused_columns False \
    \
    &> nohup.out &

echo "PreTraining started in background (PID: $!)"
echo "Monitor progress with: tail -f nohup.out"