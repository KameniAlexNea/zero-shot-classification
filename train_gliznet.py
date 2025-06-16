#!/usr/bin/env python3

import os

os.environ["WANDB_PROJECT"] = "gliznet"
os.environ["WANDB_WATCH"] = "none"

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

from gliznet.data import GliZNetDataset, collate_fn, load_dataset
from gliznet.model import GliZNetModel
from gliznet.tokenizer import GliZNETTokenizer


def main():
    @dataclass
    class ModelArgs:
        model_name: str = field(
            default="sentence-transformers/all-MiniLM-L6-v2",
            metadata={"help": "Pretrained model name or path"},
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
            default="models/fzeronet_model.pt",
            metadata={"help": "Legacy model save path"},
        )

    parser = HfArgumentParser((ModelArgs, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    # Set device
    device = (
        "cuda" if torch.cuda.is_available() and not training_args.no_cuda else "cpu"
    )
    logger.info(f"Using device: {device}")

    # Load and prepare dataset
    dataset = load_dataset(
        max_labels=model_args.max_labels,
        shuffle_labels=model_args.shuffle_labels,
    )
    splits = dataset.train_test_split(test_size=0.1, seed=42)
    train_data = splits["train"]
    val_data = splits["test"]

    logger.info(
        f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}"
    )

    # Initialize tokenizer
    tokenizer = GliZNETTokenizer.from_pretrained(model_args.model_name)

    # Create datasets
    train_dataset = GliZNetDataset(hf_dataset=train_data, tokenizer=tokenizer)

    val_dataset = GliZNetDataset(hf_dataset=val_data, tokenizer=tokenizer)

    # Initialize model
    model = GliZNetModel.from_pretrained(
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        # compute_metrics=compute_metrics,
    )

    # evaluate the model before training
    logger.info("Evaluating model before training...")
    eval_results = trainer.evaluate()
    logger.info(f"Initial evaluation results: {eval_results}")

    # Start training
    logger.info("Starting training with Transformers Trainer...")
    trainer.train()

    # Save the final model
    trainer.save_model(training_args.output_dir)

    # Also save in the legacy format for compatibility
    if model_args.save_path:
        os.makedirs(os.path.dirname(model_args.save_path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "training_args": training_args,
            },
            model_args.save_path,
        )
        logger.info(f"Legacy model saved at {model_args.save_path}")

    logger.info("Training complete")


if __name__ == "__main__":
    main()
