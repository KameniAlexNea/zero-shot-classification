#!/bin/bash

# Set environment variables
export WANDB_PROJECT="zero-shot-classification"
export WANDB_WATCH="none"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="1"
export TOKENIZERS_PARALLELISM="true"

# Training script path
SCRIPT_PATH="st_trainers/zero_shot_trainer.py"

# Model and dataset arguments
MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
DATASET_NAME="alexneakameni/ZSHOT-HARDSET"
TRAIN_SPLIT="triplet"
EVAL_SPLIT="triplet"

# Training arguments
OUTPUT_DIR="./models/zero_shot_classifier_$(date +%Y%m%d_%H%M%S)"
NUM_EPOCHS=50
BATCH_SIZE=128
MINI_BATCH_SIZE=32
LEARNING_RATE=2e-5
WARMUP_RATIO=0.1
WEIGHT_DECAY=0.01
SCALE=20.0
MARGIN=0.1
EVAL_STEPS=1000

# Run training
nohup uv run "$SCRIPT_PATH" \
    --model_name "$MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --train_split "$TRAIN_SPLIT" \
    --eval_split "$EVAL_SPLIT" \
    --mini_batch_size "$MINI_BATCH_SIZE" \
    --scale "$SCALE" \
    --margin "$MARGIN" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --per_device_eval_batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --warmup_ratio "$WARMUP_RATIO" \
    --weight_decay "$WEIGHT_DECAY" \
    --logging_steps 100 \
    --eval_strategy "epoch" \
    --eval_steps "$EVAL_STEPS" \
    --save_steps "$EVAL_STEPS" \
    --save_strategy "epoch" \
    --load_best_model_at_end \
    --metric_for_best_model "loss" \
    --report_to "wandb" \
    --dataloader_num_workers 4 \
    --fp16 \
    --run_name "zero-shot-train-${TRAIN_SPLIT}-eval-${EVAL_SPLIT}" \
    --batch_sampler no_duplicates \
    --load_best_model_at_end \
    --save_total_limit 2 \
    --lr_scheduler_type cosine \
    --eval_on_start \
    &> nohup.out &

echo "Training completed! Model saved to: $OUTPUT_DIR"
