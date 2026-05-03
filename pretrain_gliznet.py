#!/usr/bin/env python3
"""GliZNet pretraining on Exploration-Lab/COLD (846K causal/effect reasoning pairs).

This stage teaches the model to properly leverage [LAB] and [SEP] token attention
patterns on a large-scale binary choice dataset before fine-tuning on classification.
"""

import os
import warnings

import datasets
import torch
from loguru import logger
from transformers import (
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

from config.args import ModelArgs
from gliznet.augmentation import LabelAugmentationPipeline, LabelLimit
from gliznet.data import collate_fn
from gliznet.metrics import compute_metrics
from gliznet.model import GliZNetConfig, GliZNetForSequenceClassification
from gliznet.tokenizer import GliZNETTokenizer
from gliznet.training_config import GliZNetDataConfig, LabelName

os.environ["WANDB_PROJECT"] = "gliznet-pretrain"
os.environ["WANDB_WATCH"] = "none"

warnings.filterwarnings("ignore", message=".*torch._prims_common.check.*")


def create_model_tokenizer(args: ModelArgs):
    """Create GliZNet model and tokenizer from arguments."""
    tokenizer = GliZNETTokenizer.from_pretrained(
        args.model_name,
        lab_token=args.lab_cls_token,
        model_max_length=args.model_max_length,
        max_tokens_per_span=args.max_tokens_per_span,
        min_text_tokens=args.min_text_tokens,
        min_label_tokens=args.min_label_tokens,
        fix_mistral_regex=True,
    )

    if args.model_name.startswith("alexneakameni/"):
        model = GliZNetForSequenceClassification.from_pretrained(args.model_name)
        return model, tokenizer

    config = GliZNetConfig(
        backbone_model=args.model_name,
        dropout_rate=args.dropout_rate,
        focal_loss_weight=args.focal_loss_weight,
        focal_gamma=args.focal_gamma,
        supcon_loss_weight=args.supcon_loss_weight,
        label_repulsion_weight=args.label_repulsion_weight,
        supcon_margin=args.supcon_margin,
        scoring_method=args.scoring_method,
        losses=args.losses,
        lab_token_id=tokenizer.lab_token_id,
        max_labels=args.max_labels,
    )
    model = GliZNetForSequenceClassification.from_backbone_pretrained(config, tokenizer)
    logger.info(f"Model configuration: {config.to_dict()}")

    return model, tokenizer


def load_cold_dataset(seed: int = 42) -> datasets.Dataset:
    """Load Exploration-Lab/COLD and map to GliZNet format.

    Format: premise + question context → [choice1, choice2] with binary label.
    label=0 → choice1 is correct, label=1 → choice2 is correct.

    Loads all subsets: bus, cake, shopping, train, tree.
    Mapping is done on-the-fly via set_transform (no eager iteration).
    """
    logger.info("Loading Exploration-Lab/COLD dataset (all subsets)...")
    subsets = ["bus", "cake", "shopping", "train", "tree"]
    parts = []
    for subset in subsets:
        part = datasets.load_dataset("Exploration-Lab/COLD", subset, split="train")
        logger.info(f"  Loaded subset '{subset}': {len(part)} examples")
        parts.append(part)
    ds = datasets.concatenate_datasets(parts)
    logger.info(f"Raw COLD dataset size (all subsets): {len(ds)}")
    ds = ds.shuffle(seed=seed)
    logger.info(f"COLD dataset ready: {len(ds)} examples (transform applied on-the-fly)")
    return ds


def cold_transform(examples):
    """On-the-fly transform: maps COLD columns to GliZNet format."""
    question_prefix = {
        "cause": "What is the cause? ",
        "effect": "What is the effect? ",
    }
    texts = []
    ltexts = []
    lints = []
    for premise, question, choice1, choice2, label in zip(
        examples["premise"],
        examples["question"],
        examples["choice1"],
        examples["choice2"],
        examples["label"],
    ):
        prefix = question_prefix.get((question or "").strip().lower(), "")
        texts.append(prefix + (premise or "").strip())
        ltexts.append([(choice1 or "").strip(), (choice2 or "").strip()])
        label = int(label)
        lints.append([1 - label, label])

    return {
        "text": texts,
        LabelName.ltext: ltexts,
        LabelName.lint: lints,
    }


def seed_everything(seed: int = 42):
    """Set random seed for reproducibility."""
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def main():
    logger.info(f"Starting GliZNet PRETRAINING (PID: {os.getpid()})")

    parser = HfArgumentParser((ModelArgs, TrainingArguments))
    args: tuple[ModelArgs, TrainingArguments] = parser.parse_args_into_dataclasses()
    model_args, training_args = args

    seed_everything(training_args.data_seed)
    logger.info(f"Set random seed to {training_args.data_seed}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Initialize model and tokenizer
    logger.info(f"Initializing model from {model_args.model_name}...")
    model, tokenizer = create_model_tokenizer(model_args)
    logger.info(f"Model parameters: {model.num_parameters():,}")

    # Data configuration
    data_config = GliZNetDataConfig(
        max_labels=model_args.max_labels,
        shuffle_labels=model_args.shuffle_labels,
        min_label_length=model_args.min_label_length,
    )

    # Load COLD dataset
    dataset = load_cold_dataset(seed=training_args.data_seed)

    # Split into train/val
    eval_size = min(int(0.02 * len(dataset)), 5000)
    splits = dataset.train_test_split(test_size=eval_size, seed=training_args.data_seed)
    train_data = splits["train"]
    val_data = splits["test"]
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Apply COLD column mapping as a lazy transform, then tokenize on top.
    # We compose both into a single set_transform to avoid overwriting.
    logger.info("Setting up on-the-fly tokenization...")

    def make_composed_transform(shuffle_labels: bool):
        """Create a transform that maps COLD → GliZNet format → tokenized."""
        label_pipeline = LabelAugmentationPipeline(
            [LabelLimit(max_labels=data_config.max_labels, shuffle_labels=shuffle_labels)]
        )

        def composed(examples):
            # Step 1: COLD → text/ltext/lint
            mapped = cold_transform(examples)
            # Step 2: tokenize
            tokenizer_inputs = []
            labels_batch = []
            for text, label_texts, label_ints in zip(
                mapped["text"], mapped[LabelName.ltext], mapped[LabelName.lint]
            ):
                label_texts, label_ints = label_pipeline(label_texts, label_ints)
                tokenizer_inputs.append((text, label_texts))
                labels_batch.append(torch.tensor(label_ints, dtype=torch.float32))

            tokenized = tokenizer(tokenizer_inputs, return_tensors="pt")
            truncated_labels = []
            for lmask_row, label_tensor in zip(tokenized["lmask"], labels_batch):
                num_fitted = int(lmask_row.max().item()) if lmask_row.any() else 0
                truncated_labels.append(label_tensor[:num_fitted])

            return {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "lmask": tokenized["lmask"],
                "labels": truncated_labels,
            }

        return composed

    train_dataset = train_data.with_transform(make_composed_transform(shuffle_labels=data_config.shuffle_labels))
    val_dataset = val_data.with_transform(make_composed_transform(shuffle_labels=False))
    logger.info("Datasets ready (on-the-fly transform)")

    # Output directory
    os.makedirs(training_args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {training_args.output_dir}")

    # Callbacks
    callbacks = []
    if model_args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=model_args.early_stopping_patience
            )
        )

    # Trainer
    trainer = Trainer(
        model=model,
        processing_class=tokenizer.tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=callbacks,
        compute_metrics=compute_metrics,
    )

    # Log training info
    logger.info("=" * 60)
    logger.info("PRETRAINING on Exploration-Lab/COLD (846K causal reasoning)")
    logger.info(f"Total epochs: {training_args.num_train_epochs}")
    logger.info(f"Batch size per device: {training_args.per_device_train_batch_size}")
    logger.info(f"Learning rate: {training_args.learning_rate}")
    logger.info("=" * 60)

    # Save initial model
    init_path = os.path.join(training_args.output_dir, "init_model")
    os.makedirs(init_path, exist_ok=True)
    model.save_pretrained(init_path)
    tokenizer.save_pretrained(init_path)

    # Train
    try:
        trainer.train()
        logger.info("✓ Pretraining completed successfully")
    except Exception as e:
        logger.error(f"✗ Pretraining failed: {e}")
        raise

    # Evaluate
    logger.info("Evaluating on validation set...")
    eval_results = trainer.evaluate()
    logger.info(f"Eval results: {eval_results}")

    # Save pretrained model
    final_path = os.path.join(training_args.output_dir, "pretrained_model")
    os.makedirs(final_path, exist_ok=True)
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"✓ Pretrained model saved to {final_path}")


if __name__ == "__main__":
    main()
