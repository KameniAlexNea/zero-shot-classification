#!/usr/bin/env python3

import os

os.environ["WANDB_PROJECT"] = "gliznet"
os.environ["WANDB_WATCH"] = "none"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import importlib
from dataclasses import dataclass, field
from typing import Optional

import torch
from loguru import logger
from transformers import (
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

from gliznet.data import add_tokenized_function, collate_fn, load_dataset
from gliznet.metrics import compute_metrics
from gliznet.model import create_gli_znet_for_sequence_classification
from gliznet.tokenizer import GliZNETTokenizer


def get_transformers_class(class_name):
    transformers_module = importlib.import_module("transformers")
    return getattr(transformers_module, class_name)


def seed_everything(seed: int = 42):
    """Set random seed for reproducibility."""
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    @dataclass
    class ModelArgs:
        model_name: str = field(
            default="sentence-transformers/all-MiniLM-L6-v2",
            metadata={"help": "Pretrained model name or path"},
        )
        model_class: str = field(
            default="BertPreTrainedModel",
            metadata={"help": "Model class to use"},
        )
        projected_dim: int = field(
            default=256, metadata={"help": "Hidden size for projection layer"}
        )
        similarity_metric: str = field(
            default="dot",
            metadata={"help": "Similarity metric: cosine, bilinear, dot"},
        )
        max_labels: Optional[int] = field(
            default=50, metadata={"help": "Maximum number of labels"}
        )  # 50 to avoid overflow
        shuffle_labels: bool = field(default=True, metadata={"help": "Shuffle labels"})
        save_path: str = field(
            default=None,
            metadata={"help": "Legacy model save path"},
        )
        early_stopping_patience: int = field(
            default=3,
            metadata={"help": "Early stopping patience"},
        )

    parser = HfArgumentParser((ModelArgs, TrainingArguments))
    args: tuple[ModelArgs, TrainingArguments] = parser.parse_args_into_dataclasses()
    model_args, training_args = args

    # Set device
    device = (
        "cuda" if torch.cuda.is_available() and not training_args.no_cuda else "cpu"
    )
    logger.info(f"Using device: {device}")

    testing_dataset = load_dataset(split="test")

    # Load and prepare dataset
    dataset = load_dataset()
    splits = dataset.train_test_split(test_size=0.1, seed=42)
    train_data = splits["train"]
    val_data = splits["test"]

    logger.info(
        f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}"
    )

    # Initialize tokenizer
    tokenizer = GliZNETTokenizer.from_pretrained(model_args.model_name)

    # Create datasets
    train_dataset = add_tokenized_function(
        hf_dataset=train_data, tokenizer=tokenizer, max_labels=model_args.max_labels
    )

    val_dataset = add_tokenized_function(
        hf_dataset=val_data,
        tokenizer=tokenizer,
        shuffle_labels=False,
        max_labels=model_args.max_labels,
    )
    testing_dataset = add_tokenized_function(
        hf_dataset=testing_dataset,
        tokenizer=tokenizer,
        shuffle_labels=False,
        max_labels=model_args.max_labels,
    )

    # Initialize model
    pretrained_cls = create_gli_znet_for_sequence_classification(
        get_transformers_class(model_args.model_class)
    )
    model = pretrained_cls.from_pretrained(
        model_args.model_name,
        projected_dim=model_args.projected_dim,
        similarity_metric=model_args.similarity_metric,
    )

    # Create output directory
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        processing_class=tokenizer.tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=model_args.early_stopping_patience
            )
        ],
        compute_metrics=compute_metrics,
    )

    # Start training
    logger.info("Starting training with Transformers Trainer...")
    trainer.train()

    logger.info("Training complete")


if __name__ == "__main__":
    main()
