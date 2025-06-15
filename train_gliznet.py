#!/usr/bin/env python3

import argparse
import os

import torch
from loguru import logger
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from gliznet.data import GliZNetDataset, collate_fn
from gliznet.model import GliZNetModel
from gliznet.tokenizer import GliZNETTokenizer, load_dataset


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
    parser = argparse.ArgumentParser(
        description="Train GliZNetModel for zero-shot classification"
    )
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument(
        "--similarity_metric", default="bilinear", choices=["cosine", "dot", "bilinear"]
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save_path", default="models/fzeronet_model.pt")
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument(
        "--shuffle_labels",
        action="store_false",
        default=True,
        help="Disable shuffling of labels (enabled by default).",
    )
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    logger.info(f"Using device: {device}")

    # Load and prepare dataset
    dataset = load_dataset(max_labels=20, shuffle_labels=args.shuffle_labels)
    splits = dataset.train_test_split(test_size=0.1, seed=42)
    train_data = splits["train"]
    val_data = splits["test"]

    logger.info(
        f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}"
    )

    # Initialize tokenizer
    tokenizer = GliZNETTokenizer(model_name=args.model_name)

    # Create datasets
    train_dataset = GliZNetDataset(hf_dataset=train_data, tokenizer=tokenizer)

    val_dataset = GliZNetDataset(hf_dataset=val_data, tokenizer=tokenizer)

    # Initialize model
    model = GliZNetModel(
        model_name=args.model_name,
        hidden_size=args.hidden_size,
        similarity_metric=args.similarity_metric,
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        # report_to=None,  # Disable wandb/tensorboard logging
    )

    # Initialize trainer
    trainer = GliZNetTrainer(
        model=model,
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
    trainer.save_model(args.output_dir)

    # Also save in the legacy format for compatibility
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "training_args": training_args,
            },
            args.save_path,
        )
        logger.info(f"Legacy model saved at {args.save_path}")

    logger.info("Training complete")


if __name__ == "__main__":
    main()
