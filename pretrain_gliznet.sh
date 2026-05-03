#!/bin/bash
# GliZNet Pretraining Script — Exploration-Lab/COLD (846K causal reasoning)
#
# This is the first training stage. It teaches the model to properly leverage
# [LAB] and [SEP] token attention patterns on large-scale binary choice data.
#
# After pretraining, use the saved model path in train_gliznet.sh:
#   --model_name results/pretrain_cold_<TIMESTAMP>/pretrained_model
#
# Run with: bash pretrain_gliznet.sh

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results/pretrain_cold_${TIMESTAMP}"

nohup accelerate launch pretrain_gliznet.py \
    \
    `# Model Configuration` \
    --model_name microsoft/deberta-v3-base \
    --model_class DebertaV2PreTrainedModel \
    --dropout_rate 0.1 \
    \
    `# Loss Configuration — gentler for pretraining` \
    --focal_loss_weight 1.0 \
    --focal_gamma 2.0 \
    --supcon_loss_weight 0.3 \
    --label_repulsion_weight 1.0 \
    --supcon_margin 0.3 \
    --scoring_method bilinear \
    \
    `# Data Configuration` \
    --dataset_path Exploration-Lab/COLD \
    --max_labels 5 \
    --shuffle_labels \
    --min_label_length 2 \
    --data_seed 42 \
    `# Tokenizer Configuration` \
    --use_fast_tokenizer \
    --model_max_length 256 \
    --lab_cls_token "[LAB]" \
    --max_tokens_per_span 32 \
    --min_text_tokens 5 \
    --min_label_tokens 2 \
    \
    `# Training Arguments — larger LR, fewer epochs for pretraining` \
    --run_name "gliznet_pretrain_cold_${TIMESTAMP}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --warmup_steps 0.05 \
    --weight_decay 1e-2 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    \
    `# Evaluation & Checkpointing` \
    --eval_strategy steps \
    --eval_steps 2000 \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 3 \
    --load_best_model_at_end \
    --metric_for_best_model loss \
    --early_stopping_patience 5 \
    --eval_do_concat_batches False \
    \
    `# Performance Optimization` \
    --dataloader_pin_memory \
    --dataloader_num_workers 8 \
    --dataloader_prefetch_factor 2 \
    --ddp_find_unused_parameters False \
    --bf16 \
    \
    `# Logging & Monitoring` \
    --logging_steps 100 \
    --report_to wandb \
    --remove_unused_columns False \
    \
    &> nohup_pretrain.out &

echo "Pretraining started in background (PID: $!)"
echo "Output dir: ${OUTPUT_DIR}"
echo "Monitor progress with: tail -f nohup_pretrain.out"
echo ""
echo "After pretraining completes, fine-tune with:"
echo "  Edit train_gliznet.sh and set: --model_name ${OUTPUT_DIR}/pretrained_model"
