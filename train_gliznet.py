#!/usr/bin/env python3

import os
import random
import warnings

import datasets
import torch
from loguru import logger
from transformers import (
    EarlyStoppingCallback,
    HfArgumentParser,
    TrainingArguments,
)

from gliznet.config.args import ModelArgs
from gliznet.augmentation import (
    load_augmentation_pipeline,
    load_label_augmentation_pipeline,
)
from gliznet.data import add_tokenized_function, collate_fn, load_dataset
from gliznet.metrics import compute_metrics
from gliznet.model import GliZNetConfig, GliZNetForSequenceClassification
from gliznet.tokenizer import GliZNETTokenizer
from gliznet.trainer import GliZNetTrainer
from gliznet.training_config import GliZNetDataConfig
from gliznet.config.training_data import additional_datasets

os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT", "gliznet")
os.environ["WANDB_WATCH"] = "none"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

warnings.filterwarnings("ignore", message=".*torch._prims_common.check.*")


def create_model_tokenizer(args: ModelArgs):
    """Create GliZNet model and tokenizer from arguments.

    Args:
        args: ModelArgs containing model configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    # Initialize tokenizer with new API
    tokenizer = GliZNETTokenizer.from_pretrained(
        args.model_name,
        lab_token=args.lab_cls_token,
        model_max_length=args.model_max_length,
        max_tokens_per_span=args.max_tokens_per_span,
        min_text_tokens=args.min_text_tokens,
        min_label_tokens=args.min_label_tokens,
        fix_mistral_regex=True,
    )

    # Load full model (including bilinear head) if resuming from a saved checkpoint
    if os.path.isdir(args.model_name) or args.model_name.startswith("alexneakameni/"):
        config = GliZNetConfig.from_pretrained(
            args.model_name,
            dropout_rate=args.dropout_rate,
            focal_loss_weight=args.focal_loss_weight,
            focal_gamma=args.focal_gamma,
            supcon_loss_weight=args.supcon_loss_weight,
            label_repulsion_weight=args.label_repulsion_weight,
            alignment_loss_weight=args.alignment_loss_weight,
            supcon_margin=args.supcon_margin,
            scoring_method=args.scoring_method,
            losses=args.losses,
            lab_token_id=tokenizer.lab_token_id,
            max_labels=args.max_labels,
            enrich_labels=args.enrich_labels,
        )
        model = GliZNetForSequenceClassification.from_pretrained(
            args.model_name, config=config
        )
        logger.info(f"Loaded full model from {args.model_name}")
        return model, tokenizer

    # Create GliZNet configuration
    config = GliZNetConfig(
        backbone_model=args.model_name,
        dropout_rate=args.dropout_rate,
        enrich_labels=args.enrich_labels,
        # Loss configuration
        focal_loss_weight=args.focal_loss_weight,
        focal_gamma=args.focal_gamma,
        supcon_loss_weight=args.supcon_loss_weight,
        label_repulsion_weight=args.label_repulsion_weight,
        alignment_loss_weight=args.alignment_loss_weight,
        supcon_margin=args.supcon_margin,
        scoring_method=args.scoring_method,
        losses=args.losses,
        # label id
        lab_token_id=tokenizer.lab_token_id,
        max_labels=args.max_labels,
    )
    model = GliZNetForSequenceClassification.from_backbone_pretrained(config, tokenizer)
    logger.info(f"Model configuration: {config.to_dict()}")

    return model, tokenizer


def sample_dataset(ds: datasets.Dataset, max_size: int = 50_000):
    if len(ds) < max_size:
        return ds
    index = list(range(len(ds)))
    rand = random.Random(42)
    rand.shuffle(index)
    return ds.select(index[:max_size])


def add_additional_ds(
    base_ds: datasets.Dataset, max_size: int = 50_000, seed: int = 42
):
    ds = datasets.concatenate_datasets(
        [base_ds]
        + [ds_loader(max_size, seed) for ds_loader in additional_datasets.values()]
    )
    return ds


def seed_everything(seed: int = 42):
    """Set random seed for reproducibility."""
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def main():
    logger.info(f"Starting GliZNet training process (PID: {os.getpid()})")

    parser = HfArgumentParser((ModelArgs, TrainingArguments))
    args: tuple[ModelArgs, TrainingArguments] = parser.parse_args_into_dataclasses()
    model_args, training_args = args

    # Set random seeds for reproducibility
    seed_everything(training_args.data_seed)
    logger.info(f"Set random seed to {training_args.data_seed}")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Validate configuration
    logger.info("Validating configuration...")
    logger.info(f"Model: {model_args.model_name}")
    logger.info(f"Label separator token: {model_args.lab_cls_token}")
    logger.info(f"Max sequence length: {model_args.model_max_length}")

    # Initialize model and tokenizer
    logger.info(f"Initializing model and tokenizer from {model_args.model_name}...")
    model, tokenizer = create_model_tokenizer(model_args)

    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
    logger.info(f"Model parameters: {model.num_parameters():,}")

    # Create data configuration
    data_config = GliZNetDataConfig(
        max_labels=model_args.max_labels,
        shuffle_labels=model_args.shuffle_labels,
        min_label_length=model_args.min_label_length,
    )
    logger.info(f"Data config: {data_config}")

    # Load datasets
    logger.info(f"Loading dataset from {model_args.dataset_path}...")
    # Final benchmark evaluation set
    testing_data = load_dataset(
        path="alexneakameni/eval-zero-shot-classification",
        split="test",
        min_label_length=data_config.min_label_length,
    )

    # Validation: ZSHOT-HARDSET-v2 test split
    val_data = load_dataset(
        path="alexneakameni/ZSHOT-HARDSET-v2",
        name=model_args.dataset_name,
        split="test",
        min_label_length=data_config.min_label_length,
    )

    # Training: ZSHOT-HARDSET-v2 train split + optional additional datasets
    train_data_base = load_dataset(
        path=model_args.dataset_path,
        name=model_args.dataset_name,
        split="train",
        min_label_length=data_config.min_label_length,
    )
    train_data_fixed = load_dataset(
        path=model_args.dataset_path,
        name=model_args.dataset_name,
        split="train_fixed",
        min_label_length=data_config.min_label_length,
    )
    train_data = datasets.concatenate_datasets([train_data_base, train_data_fixed])
    train_data.shuffle(seed=training_args.data_seed, writer_batch_size=50000)
    size_before = len(train_data)
    if model_args.use_additional_datasets and model_args.max_extended_ds_size > 0:
        train_data = add_additional_ds(
            train_data, model_args.max_extended_ds_size, training_args.data_seed
        )
    added_size = len(train_data) - size_before

    logger.info(
        f"Dataset loaded - Train: {len(train_data)} with {added_size} added, Val: {len(val_data)}, Test: {len(testing_data)}"
    )

    # Create datasets (note: token_dropout removed, should be in collate_fn if needed)
    logger.info("Tokenizing datasets...")

    # Load text augmentation pipeline if enabled
    aug_pipeline = None
    if model_args.text_augmentation:
        aug_pipeline = load_augmentation_pipeline(model_args.augmentation_config)
        logger.info(f"Text augmentation enabled: {aug_pipeline!r}")

    # Load label augmentation pipeline from config
    train_label_pipeline = load_label_augmentation_pipeline(
        config_path=model_args.augmentation_config,
        max_labels=data_config.max_labels,
    )
    logger.info(f"Train label augmentation: {train_label_pipeline!r}")

    train_dataset = add_tokenized_function(
        hf_dataset=train_data,
        tokenizer=tokenizer,
        max_labels=data_config.max_labels,
        shuffle_labels=data_config.shuffle_labels,
        as_transform=True,
        augmentation_pipeline=aug_pipeline,
        label_augmentation_pipeline=train_label_pipeline,
    )

    val_dataset = add_tokenized_function(
        hf_dataset=val_data,
        tokenizer=tokenizer,
        shuffle_labels=False,  # Don't shuffle validation labels
        max_labels=data_config.max_labels,
        as_transform=True,
    )

    testing_dataset = add_tokenized_function(
        hf_dataset=testing_data,
        tokenizer=tokenizer,
        shuffle_labels=False,  # Don't shuffle test labels
        max_labels=data_config.max_labels,
        as_transform=True,
    )
    logger.info("Datasets tokenized successfully")

    # Log model configuration
    logger.info(f"Model initialized - Parameters: {model.num_parameters():,}")
    logger.info(f"Model config: {model.config}")

    # Create output directory
    os.makedirs(training_args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {training_args.output_dir}")

    # Configure metrics computation
    metrics = compute_metrics

    # Initialize trainer
    logger.info("Initializing Trainer...")
    callbacks = []
    if model_args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=model_args.early_stopping_patience
            )
        )
        logger.info(
            f"Early stopping enabled with patience={model_args.early_stopping_patience}"
        )

    trainer = GliZNetTrainer(
        model=model,
        processing_class=tokenizer.tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=callbacks,
        compute_metrics=metrics,
    )

    # Start training
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info(f"Total epochs: {training_args.num_train_epochs}")
    logger.info(f"Batch size per device: {training_args.per_device_train_batch_size}")
    logger.info(f"Learning rate: {training_args.learning_rate}")
    logger.info("=" * 60)

    # Save final model
    final_model_path = os.path.join(training_args.output_dir, "init_model")
    logger.info(f"Saving initial model to {final_model_path}...")
    os.makedirs(final_model_path, exist_ok=True)
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info("✓ Model and tokenizer saved successfully")

    try:
        trainer.train()
        logger.info("✓ Training completed successfully")
    except Exception as e:
        logger.error(f"✗ Training failed: {e}")
        raise

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(testing_dataset, metric_key_prefix="test")
    logger.info(f"Test results: {test_results}")
    logger.info("✓ Evaluation complete")

    # Save final model
    final_model_path = os.path.join(training_args.output_dir, "final_model")
    logger.info(f"Saving final model to {final_model_path}...")
    os.makedirs(final_model_path, exist_ok=True)
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info("✓ Model and tokenizer saved successfully")


if __name__ == "__main__":
    main()
