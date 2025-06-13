#!/usr/bin/env python3

import argparse
from typing import Dict, Optional

import torch
import torch.optim as optim
from loguru import logger

from gliznet.tokenizer import GliZNETTokenizer, load_dataset
from gliznet.model import GliZNetModel
from gliznet.data import prepare_data_loaders


def train_model(
    model: GliZNetModel,
    train_loader,
    val_loader,
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
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}"
                )

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
    parser = argparse.ArgumentParser(
        description="Train GliZNetModel for zero-shot classification"
    )
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument(
        "--similarity_metric", default="cosine", choices=["cosine", "dot", "bilinear"]
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save_path", default="fzeronet_model.pt")
    parser.add_argument("--data_path")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--shuffle_labels", action="store_true", default=True)

    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    logger.info(f"Using device: {device}")

    dataset = load_dataset(max_labels=50)

    splits = dataset.train_test_split(test_size=0.1, seed=42)
    train_data = splits["train"]
    val_data = splits["test"]

    logger.info(
        f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}"
    )

    tokenizer = GliZNETTokenizer(model_name=args.model_name)

    # Create DataLoaders using the new efficient approach
    train_loader, val_loader = prepare_data_loaders(
        train_dataset=train_data,
        val_dataset=val_data,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
    )

    model = GliZNetModel(
        model_name=args.model_name,
        hidden_size=args.hidden_size,
        similarity_metric=args.similarity_metric,
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    _ = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        args.num_epochs,
        device,
        args.save_path,
    )
    logger.info("Training complete")


if __name__ == "__main__":
    main()
