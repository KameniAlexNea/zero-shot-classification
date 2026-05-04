#!/bin/bash
# GliZNet Continued Pretraining — pszemraj/unified-mcqa-all (683K multi-choice QA)
#
# Stage 2: Continue from COLD checkpoint on longer, multi-label data.
# This teaches the model to handle 4-5 long labels competing for [LAB] attention.
#
# Run with: bash pretrain_mcqa.sh

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results/pretrain_mcqa_${TIMESTAMP}"

# Resume from COLD pretrained checkpoint
MODEL_PATH="results/pretrain_cold_20260503_180416/checkpoint-40000"

nohup accelerate launch pretrain_gliznet.py \
    \
    `# Model Configuration — resume from COLD checkpoint` \
    --model_name "${MODEL_PATH}" \
    --model_class DebertaV2PreTrainedModel \
    --dropout_rate 0.1 \
    \
    `# Loss Configuration` \
    --focal_loss_weight 0.5 \
    --focal_gamma 2.0 \
    --supcon_loss_weight 0.5 \
    --label_repulsion_weight 5.0 \
    --supcon_margin 0.5 \
    --scoring_method bilinear \
    \
    `# Data Configuration` \
    --dataset_path pszemraj/unified-mcqa-all \
    --max_labels 10 \
    --shuffle_labels \
    --min_label_length 2 \
    --data_seed 42 \
    `# Tokenizer Configuration — longer seq for long choices` \
    --use_fast_tokenizer \
    --model_max_length 512 \
    --lab_cls_token "[LAB]" \
    --max_tokens_per_span 64 \
    --min_text_tokens 10 \
    --min_label_tokens 3 \
    \
    `# Training Arguments — conservative LR for continued pretraining` \
    --run_name "gliznet_pretrain_mcqa_${TIMESTAMP}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --warmup_steps 0.05 \
    --weight_decay 1e-2 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    \
    `# Evaluation & Checkpointing` \
    --eval_strategy steps \
    --eval_steps 500 \
    --save_strategy steps \
    --save_steps 500 \
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
    --logging_steps 50 \
    --report_to wandb \
    --remove_unused_columns False \
    \
    &> nohup_pretrain_mcqa.out &

echo "Continued pretraining started in background (PID: $!)"
echo "Output dir: ${OUTPUT_DIR}"
echo "Resuming from: ${MODEL_PATH}"
echo "Monitor progress with: tail -f nohup_pretrain_mcqa.out"
echo ""
echo "After completion, fine-tune with:"
echo "  --model_name ${OUTPUT_DIR}/pretrained_model"
