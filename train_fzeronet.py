#!/usr/bin/env python3
"""
Training script for FZeroNet - Zero-shot Classification Model
"""

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from typing import List, Dict, Optional
import argparse
from pathlib import Path
from datasets import Dataset as HFDataset, load_dataset

from gliznet.tokenizer import ZeroShotClassificationTokenizer
from gliznet.model import FZeroNet
from loguru import logger


class ZeroShotDataset(Dataset):
    """
    Dataset class for zero-shot classification training using HuggingFace datasets.
    """

    def __init__(
        self,
        dataset: HFDataset,
        tokenizer: ZeroShotClassificationTokenizer,
        text_column: str = "text",
        positive_labels_column: str = "positive_labels",
        negative_labels_column: str = "negative_labels",
        shuffle_labels: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            dataset: HuggingFace Dataset with text and labels
            tokenizer: Tokenizer instance
            text_column: Name of the column containing text
            positive_labels_column: Name of the column containing positive labels
            negative_labels_column: Name of the column containing negative labels
            shuffle_labels: Whether to shuffle the order of labels to avoid bias
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.positive_labels_column = positive_labels_column
        self.negative_labels_column = negative_labels_column
        self.shuffle_labels = shuffle_labels

        logger.info(f"Dataset initialized with {len(dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict:
        item = self.dataset[idx]

        text = item[self.text_column]
        positive_labels = item[self.positive_labels_column]
        negative_labels = item[self.negative_labels_column]

        # Create combined label list and ground truth
        all_labels = positive_labels + negative_labels
        ground_truth = [1.0] * len(positive_labels) + [0.0] * len(negative_labels)

        # Shuffle labels and ground truth together to avoid positional bias
        if self.shuffle_labels:
            combined = list(zip(all_labels, ground_truth))
            random.shuffle(combined)
            all_labels, ground_truth = zip(*combined)
            all_labels = list(all_labels)
            ground_truth = list(ground_truth)

        # Tokenize
        tokenized = self.tokenizer.tokenize_example(
            text, all_labels, return_tensors="pt"
        )

        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "label_mask": tokenized["label_mask"].squeeze(0),
            "labels": torch.tensor(ground_truth, dtype=torch.float32),
            "text": text,
            "all_labels": all_labels,
        }


def collate_fn(batch):
    """Custom collate function to handle variable-length labels."""
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "label_mask": torch.stack([item["label_mask"] for item in batch]),
        "labels": [item["labels"] for item in batch],  # Keep as list for variable sizes
        "texts": [item["text"] for item in batch],
        "all_labels": [item["all_labels"] for item in batch],
    }


def train_model(
    model: FZeroNet,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    save_path: Optional[str] = None,
) -> Dict:
    """
    Train the FZeroNet model.

    Args:
        model: FZeroNet model instance
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        optimizer: Optimizer instance
        num_epochs: Number of training epochs
        device: Device to train on
        save_path: Path to save the model (optional)

    Returns:
        Training history dictionary
    """
    model.to(device)

    train_losses = []
    val_losses = []

    logger.info(f"Starting training for {num_epochs} epochs on {device}")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_mask = batch["label_mask"].to(device)
            labels = [
                label.to(device) for label in batch["labels"]
            ]  # Variable size labels

            # Forward pass
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                label_mask=label_mask,
                labels=labels,
            )

            loss = outputs["loss"]

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}"
                )

        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)

        # Validation phase
        if val_loader is not None:
            model.eval()
            epoch_val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    # Move batch to device
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    label_mask = batch["label_mask"].to(device)
                    labels = [
                        label.to(device) for label in batch["labels"]
                    ]  # Variable size labels

                    # Forward pass
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        label_mask=label_mask,
                        labels=labels,
                    )

                    loss = outputs["loss"]
                    epoch_val_loss += loss.item()
                    val_batches += 1

                    # Calculate accuracy - simplified for variable sizes
                    # We'll just track the loss for now

            avg_val_loss = epoch_val_loss / val_batches
            val_losses.append(avg_val_loss)

            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}"
            )
        else:
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}"
            )

    # Save model if path provided
    if save_path:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
            },
            save_path,
        )
        logger.info(f"Model saved to {save_path}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train FZeroNet for zero-shot classification"
    )
    parser.add_argument(
        "--model_name", default="bert-base-uncased", help="HuggingFace model name"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=256, help="Hidden size for projections"
    )
    parser.add_argument(
        "--similarity_metric", default="cosine", choices=["cosine", "dot", "bilinear"]
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--num_epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="Learning rate"
    )
    parser.add_argument(
        "--device", default="auto", help="Device to use (auto, cpu, cuda)"
    )
    parser.add_argument(
        "--save_path", default="fzeronet_model.pt", help="Path to save the model"
    )
    parser.add_argument(
        "--data_path",
        help="Path to training data JSON file or HuggingFace dataset name",
    )
    parser.add_argument("--text_column", default="text", help="Name of text column")
    parser.add_argument(
        "--positive_labels_column",
        default="positive_labels",
        help="Name of positive labels column",
    )
    parser.add_argument(
        "--negative_labels_column",
        default="negative_labels",
        help="Name of negative labels column",
    )
    parser.add_argument(
        "--shuffle_labels",
        action="store_true",
        default=True,
        help="Shuffle label order to avoid bias",
    )

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Load or create dataset
    if args.data_path:
        if Path(args.data_path).exists():
            # Load from local JSON file
            dataset = load_dataset("json", data_files=args.data_path)["train"]
        else:
            # Try to load from HuggingFace Hub
            dataset = load_dataset(args.data_path)["train"]
    else:
        # Create sample dataset using HuggingFace dataset format
        sample_data = [
            {
                "text": "Scientists discovered a new protein structure using AI.",
                "positive_labels": ["science", "research", "artificial_intelligence"],
                "negative_labels": ["sports", "cooking", "travel"],
            },
            {
                "text": "The basketball team won the championship.",
                "positive_labels": ["sports", "basketball", "competition"],
                "negative_labels": ["science", "cooking", "technology"],
            },
        ]
        dataset = HFDataset.from_list(sample_data)

    # Split dataset
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset_hf = dataset["train"]
    val_dataset_hf = dataset["test"]

    logger.info(f"Training samples: {len(train_dataset_hf)}")
    logger.info(f"Validation samples: {len(val_dataset_hf)}")

    # Initialize tokenizer and model
    tokenizer = ZeroShotClassificationTokenizer(model_name=args.model_name)
    model = FZeroNet(
        model_name=args.model_name,
        hidden_size=args.hidden_size,
        similarity_metric=args.similarity_metric,
    )

    # Create datasets and data loaders
    train_dataset = ZeroShotDataset(
        train_dataset_hf,
        tokenizer,
        text_column=args.text_column,
        positive_labels_column=args.positive_labels_column,
        negative_labels_column=args.negative_labels_column,
        shuffle_labels=args.shuffle_labels,
    )
    val_dataset = ZeroShotDataset(
        val_dataset_hf,
        tokenizer,
        text_column=args.text_column,
        positive_labels_column=args.positive_labels_column,
        negative_labels_column=args.negative_labels_column,
        shuffle_labels=args.shuffle_labels,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Train model
    _ = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        device=device,
        save_path=args.save_path,
    )

    # Evaluate on validation set
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
