# GliZNet Training Configuration Guide

This directory contains example configurations for training GliZNet models.

## Configuration Files

- `example_config.json` - Complete training configuration with separator pooling
- `mean_pooling_config.json` - Configuration for mean pooling strategy (coming soon)
- `small_model_config.json` - Lightweight config for quick experiments (coming soon)

## Using Configurations

### Method 1: Command Line Arguments (Recommended)

```bash
bash train_gliznet.sh
```

The shell script contains all parameters with clear comments.

### Method 2: Python Script with Custom Config

```python
from gliznet.config import GliZNetTrainingConfig, GliZNetDataConfig

# Create custom configuration
training_config = GliZNetTrainingConfig(
    projected_dim=512,
    similarity_metric="dot",
    use_separator_pooling=True,
)

data_config = GliZNetDataConfig(
    max_labels=30,
    shuffle_labels=True,
)

# Pass to training
# ... (see examples.py for full usage)
```

## Key Configuration Options

### Pooling Strategy

**Separator Pooling (Recommended for new models):**
```bash
--use_separator_pooling \
--cls_separator_token "[LAB]"
```
- Uses separator token embeddings directly
- Requires adding custom [LAB] token to vocabulary
- More parameter-efficient
- Better for models with custom tokens

**Mean Pooling (Default):**
```bash
--cls_separator_token ";"
```
- Averages label content token embeddings
- Works with any tokenizer without modification
- More flexible for different label formats

### Loss Components

**BCE Loss (Primary):**
```bash
--scale_loss 10.0
```
Binary cross-entropy between predictions and labels.

**Contrastive Loss (Hard Negative Mining):**
```bash
--margin 0.1 \
--contrastive_loss_weight 1.0 \
--temperature 1.0
```
Separates positive and negative labels using hardest examples.

### Similarity Metrics

- `dot`: Simple dot product (fastest)
- `dot_learning`: Learned linear projection of element-wise product
- `bilinear`: Full bilinear transformation (most expressive but slowest)

### Data Configuration

```bash
--max_labels 20           # Maximum labels per sample
--shuffle_labels          # Randomly shuffle labels (maintains proportion)
--min_label_length 2      # Minimum label text length
--data_seed 42            # Random seed for reproducibility
```

## Quick Start Recipes

### High Performance Setup (GPU with 16GB+ VRAM)
```bash
--per_device_train_batch_size 128 \
--fp16 \
--auto_find_batch_size \
--dataloader_num_workers 8
```

### Memory-Constrained Setup
```bash
--per_device_train_batch_size 32 \
--gradient_accumulation_steps 4 \
--fp16 \
--dataloader_num_workers 2
```

### Quick Experimentation
```bash
--num_train_epochs 1 \
--max_labels 10 \
--eval_steps 0.5 \
--logging_steps 50
```

## Monitoring Training

### WandB Integration
```bash
--report_to wandb \
--run_name "my_experiment"
```

### TensorBoard
```bash
--report_to tensorboard \
--logging_dir ./logs
```

### Console Only
```bash
--report_to none
```

## Troubleshooting

### Out of Memory
1. Reduce `per_device_train_batch_size`
2. Reduce `max_labels`
3. Reduce `projected_dim`
4. Enable `gradient_checkpointing_enable` in model

### Slow Training
1. Increase `dataloader_num_workers`
2. Enable `dataloader_pin_memory`
3. Use `fp16` training
4. Set `dataloader_prefetch_factor` to 2-4

### Poor Convergence
1. Adjust `learning_rate` (try 5e-5 to 2e-4)
2. Increase `warmup_ratio` to 0.05-0.1
3. Adjust loss weights (`contrastive_loss_weight`, `scale_loss`)
4. Try different `similarity_metric`

## Best Practices

1. **Always set a seed** for reproducibility
2. **Use separator pooling** for new models (better performance)
3. **Start with small max_labels** (10-20) and increase if needed
4. **Monitor all loss components** in WandB to understand training dynamics
5. **Use early stopping** to prevent overfitting
6. **Validate pooling strategy** matches tokenizer configuration

## Example Workflows

### Training from Scratch
```bash
# 1. Edit train_gliznet.sh with your parameters
# 2. Run training
bash train_gliznet.sh

# 3. Monitor progress
tail -f nohup.out
```

### Hyperparameter Tuning
```bash
# Run multiple experiments with different configs
for lr in 5e-5 1e-4 2e-4; do
    python train_gliznet.py \
        --learning_rate $lr \
        --run_name "gliznet_lr_${lr}" \
        --output_dir "results/lr_${lr}"
done
```

### Resume Training
```bash
python train_gliznet.py \
    --resume_from_checkpoint results/checkpoint-1000 \
    # ... other args
```
