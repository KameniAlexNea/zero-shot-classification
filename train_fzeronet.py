#!/usr/bin/env python3

import random
import argparse
from pathlib import Path
from typing import Dict, Optional, List

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset, load_dataset
from loguru import logger

from gliznet.tokenizer import GliZNETTokenizer
from gliznet.model import GliZNetModel


class ZeroShotDataset(Dataset):
    def __init__(
        self,
        dataset: HFDataset,
        tokenizer: GliZNETTokenizer,
        text_column: str = "text",
        positive_labels_column: str = "positive_labels",
        negative_labels_column: str = "negative_labels",
        shuffle_labels: bool = True,
    ):
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
        pos_labels = item[self.positive_labels_column]
        neg_labels = item[self.negative_labels_column]

        all_labels = pos_labels + neg_labels
        targets = [1.0] * len(pos_labels) + [0.0] * len(neg_labels)

        if self.shuffle_labels:
            combined = list(zip(all_labels, targets))
            random.shuffle(combined)
            all_labels, targets = zip(*combined)

        tokenized = self.tokenizer.tokenize_example(text, all_labels, return_tensors="pt")

        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "label_mask": tokenized["label_mask"].squeeze(0),
            "labels": torch.tensor(targets, dtype=torch.float32),
            "text": text,
            "all_labels": list(all_labels),
        }


def collate_fn(batch: List[Dict]):
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "label_mask": torch.stack([item["label_mask"] for item in batch]),
        "labels": [item["labels"] for item in batch],
        "texts": [item["text"] for item in batch],
        "all_labels": [item["all_labels"] for item in batch],
    }


def train_model(
    model: GliZNetModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    save_path: Optional[str] = None,
) -> Dict:
    model.to(device)
    train_losses, val_losses = [], []
    logger.info(f"Starting training on {device} for {num_epochs} epochs")

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_mask = batch["label_mask"].to(device)
            labels = [label.to(device) for label in batch["labels"]]

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, label_mask, labels)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        logger.info(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f}")

        if val_loader is not None:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    label_mask = batch["label_mask"].to(device)
                    labels = [label.to(device) for label in batch["labels"]]

                    outputs = model(input_ids, attention_mask, label_mask, labels)
                    val_loss += outputs["loss"].item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            logger.info(f"Epoch {epoch} - Val Loss: {avg_val_loss:.4f}")

    if save_path:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
            },
            save_path,
        )
        logger.info(f"Model saved at {save_path}")

    return {"train_losses": train_losses, "val_losses": val_losses}


def main():
    parser = argparse.ArgumentParser(description="Train GliZNetModel for zero-shot classification")
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--similarity_metric", default="cosine", choices=["cosine", "dot", "bilinear"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save_path", default="fzeronet_model.pt")
    parser.add_argument("--data_path")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--positive_labels_column", default="positive_labels")
    parser.add_argument("--negative_labels_column", default="negative_labels")
    parser.add_argument("--shuffle_labels", action="store_true", default=True)

    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    logger.info(f"Using device: {device}")

    if args.data_path:
        data_path = Path(args.data_path)
        dataset = load_dataset("json", data_files=str(data_path))["train"] if data_path.exists() else load_dataset(args.data_path)["train"]
    else:
        dataset = HFDataset.from_list([
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
        ])

    splits = dataset.train_test_split(test_size=0.2, seed=42)
    train_data = splits["train"]
    val_data = splits["test"]

    logger.info(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}")

    tokenizer = GliZNETTokenizer(model_name=args.model_name)
    model = GliZNetModel(
        model_name=args.model_name,
        hidden_size=args.hidden_size,
        similarity_metric=args.similarity_metric,
    )

    train_dataset = ZeroShotDataset(
        train_data,
        tokenizer,
        args.text_column,
        args.positive_labels_column,
        args.negative_labels_column,
        shuffle_labels=args.shuffle_labels,
    )
    val_dataset = ZeroShotDataset(
        val_data,
        tokenizer,
        args.text_column,
        args.positive_labels_column,
        args.negative_labels_column,
        shuffle_labels=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    _ = train_model(model, train_loader, val_loader, optimizer, args.num_epochs, device, args.save_path)
    logger.info("Training complete")


if __name__ == "__main__":
    main()
