"""
Efficient DataLoader implementation for GliZNet training.

This module provides PyTorch DataLoader-based data loading for improved efficiency
compared to the original HuggingFace datasets approach.
"""

from typing import List, Dict, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import datasets
from gliznet.tokenizer import GliZNETTokenizer


class GliZNetDataset(Dataset):
    """
    PyTorch Dataset for GliZNet that handles tokenization and batching efficiently.
    """

    def __init__(
        self,
        hf_dataset: datasets.Dataset,
        tokenizer: GliZNETTokenizer,
        text_column: str = "text",
        labels_text_column: str = "labels_text",
        labels_int_column: str = "labels_int",
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.labels_text_column = labels_text_column
        self.labels_int_column = labels_int_column

        # Cache the dataset length
        self._length = len(hf_dataset)

        # Create indices for shuffling
        self.indices = list(range(self._length))
        self.current_epoch = 0

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset."""

        # Get the raw example
        example = self.hf_dataset[idx]

        # Tokenize the example
        tokenized = self.tokenizer(
            example[self.text_column],
            example[self.labels_text_column],
            return_tensors="pt",
            pad=True,
        )

        # Convert labels to tensor
        labels_int = example[self.labels_int_column]
        if isinstance(labels_int, list):
            labels = torch.tensor(labels_int, dtype=torch.float32)
        else:
            labels = torch.tensor([labels_int], dtype=torch.float32)

        # Remove batch dimension from tokenized outputs (DataLoader will add it back)
        result = {
            "input_ids": tokenized["input_ids"].unsqueeze(0),
            "attention_mask": tokenized["attention_mask"].unsqueeze(0),
            "label_mask": tokenized["label_mask"].unsqueeze(0),
            "labels": labels.unsqueeze(0),
        }

        return result


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # Stack regular tensors
    input_ids = torch.cat([item["input_ids"] for item in batch])
    attention_mask = torch.cat([item["attention_mask"] for item in batch])
    label_mask = torch.cat([item["label_mask"] for item in batch])

    # Handle labels which can have different lengths per sample
    labels = [item["labels"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_mask": label_mask,
        "labels": labels,
    }


def create_dataloader(
    hf_dataset: datasets.Dataset,
    tokenizer: GliZNETTokenizer,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    text_column: str = "text",
    labels_text_column: str = "labels_text",
    labels_int_column: str = "labels_int",
    **kwargs
) -> DataLoader:
    dataset = GliZNetDataset(
        hf_dataset=hf_dataset,
        tokenizer=tokenizer,
        text_column=text_column,
        labels_text_column=labels_text_column,
        labels_int_column=labels_int_column,
        shuffle_on_epoch=shuffle,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # We handle shuffling in the dataset
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        **kwargs
    )


def prepare_data_loaders(
    train_dataset: datasets.Dataset,
    val_dataset: Optional[datasets.Dataset],
    tokenizer: GliZNETTokenizer,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    **kwargs
) -> Tuple[DataLoader, Optional[DataLoader]]:
    train_loader = create_dataloader(
        train_dataset,
        tokenizer,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = create_dataloader(
            val_dataset,
            tokenizer,
            batch_size=batch_size,
            shuffle=False,  # No shuffling for validation
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs
        )

    return train_loader, val_loader
