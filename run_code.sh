# export WANDB_PROJECT="zero-shot-classification"
# export WANDB_WATCH="none"
# export CUDA_DEVICE_ORDER="PCI_BUS_ID"

python mteb_style_evals.py \
    --model_path "results/small/checkpoint-1966" \
    --data "poem_sentiment" \
    --classifier_type "knn" \
    --k 5 \
    --batch_size 128 \
    --device_pos 0 \
    --model_class "DebertaV2PreTrainedModel"