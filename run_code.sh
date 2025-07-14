export WANDB_PROJECT="zero-shot-classification"
export WANDB_WATCH="none"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

python mteb_style_evals.py \
    --model_path "results/experiments/gliznet/gliznet-small/clip-gliner/checkpoint-3932" \
    --data "amazon_massive_intent" \
    --classifier_type "knn" \
    --k 5 \
    --batch_size 128 \
    --device_pos 0 \
    --model_class "DebertaV2PreTrainedModel"