#!/usr/bin/env python3

import os
os.environ["WANDB_PROJECT"] = "gliznet"
os.environ["WANDB_WATCH"] = "none"

import torch
from loguru import logger
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, HfArgumentParser

from gliznet.data import GliZNetDataset, collate_fn
from gliznet.model import GliZNetModel
from gliznet.tokenizer import GliZNETTokenizer, load_dataset

from dataclasses import dataclass, field
from typing import Optional


class GliZNetTrainer(Trainer):
    """Custom Trainer for GliZNetModel that handles the specific data format."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        """Compute loss for GliZNetModel."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        label_mask = inputs["label_mask"]
        labels = inputs["labels"]

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            label_mask=label_mask,
            labels=labels,
        )

        loss = outputs.get("loss")
        if loss is None:
            # If no loss computed (no valid labels), return zero loss
            loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)

        return (loss, outputs) if return_outputs else loss


def main():
    @dataclass
    class ModelArgs:
        model_name: str = field(default="bert-base-uncased", metadata={"help": "Pretrained model name or path"})
        hidden_size: int = field(default=256, metadata={"help": "Hidden size for projection layer"})
        similarity_metric: str = field(default="bilinear", metadata={"help": "Similarity metric"})
        max_labels: Optional[int] = field(default=None, metadata={"help": "Maximum number of labels"})
        shuffle_labels: bool = field(default=True, metadata={"help": "Shuffle labels"})
        save_path: str = field(default="models/fzeronet_model.pt", metadata={"help": "Legacy model save path"})

    parser = HfArgumentParser((ModelArgs, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    # Set device
    device = "cuda" if torch.cuda.is_available() and not training_args.no_cuda else "cpu"
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
    model = GliZNetModel(
        model_name=model_args.model_name,
        hidden_size=model_args.hidden_size,
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
    )

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
