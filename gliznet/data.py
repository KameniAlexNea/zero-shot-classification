"""
Efficient DataLoader implementation for GliZNet training.

This module provides PyTorch DataLoader-based data loading for improved efficiency
compared to the original HuggingFace datasets approach.
"""

import random
from typing import Dict, List

import datasets
import torch
from torch.utils.data import Dataset

from gliznet.tokenizer import GliZNETTokenizer


def load_dataset(
    path: str = "alexneakameni/ZSHOT-HARDSET",
    name: str = "triplet",
    split: str = "train",
    text_column: str = "sentence",
    positive_column: str = "labels",
    negative_column: str = "not_labels",
    shuffle_labels: bool = True,
):
    def mapper(x):
        labels = x[positive_column] + x[negative_column]
        labels_int = [1] * len(x[positive_column]) + [0] * len(x[negative_column])
        if shuffle_labels:
            combined = list(zip(labels, labels_int))
            random.shuffle(combined)
            labels, labels_int = zip(*combined)
        return {
            "text": x[text_column],
            "labels_text": labels,
            "labels_int": labels_int,
        }

    ds = datasets.load_dataset(path, name)[split]
    ds = ds.map(
        mapper,
    )
    return ds.select_columns(["text", "labels_text", "labels_int"])


def add_tokenized_function(
    hf_dataset: datasets.Dataset,
    tokenizer: GliZNETTokenizer,
    text_column: str = "text",
    labels_text_column: str = "labels_text",
    labels_int_column: str = "labels_int",
    max_labels=50,
    shuffle_labels: bool = True,
) -> datasets.Dataset:
    """
    Tokenizes the HuggingFace dataset using the GliZNETTokenizer.
    """

    def tokenize_function(examples):
        text = examples[text_column]
        labels_text = examples[labels_text_column]
        labels_int = examples[labels_int_column]

        # randomly shuffle labels if they are more than max_labels
        if isinstance(labels_text, list) and len(labels_text) > max_labels:
            combined = list(zip(labels_text, labels_int))
            if shuffle_labels:
                random.shuffle(combined)
            labels_text, labels_int = zip(*combined[:max_labels])
            labels_int = list(labels_int)  # Convert back to list

        # Tokenize the example
        tokenized: dict[str, torch.Tensor] = tokenizer(
            text,
            labels_text,
            return_tensors="pt",
            pad=True,
        )

        # Convert labels to tensor
        if isinstance(labels_int, list):
            labels = torch.tensor(labels_int, dtype=torch.float32)
        else:
            labels = torch.tensor([labels_int], dtype=torch.float32)

        # Return without adding batch dimension (DataLoader will handle batching)
        result = {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "label_mask": tokenized["label_mask"].squeeze(0),
            "labels": labels.reshape(1, -1),  # Ensure labels are 2D
        }

        return result

    return hf_dataset.with_transform(
        tokenize_function,
    )


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
        max_labels=50,
        shuffle_labels: bool = True,
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.labels_text_column = labels_text_column
        self.labels_int_column = labels_int_column
        self.max_labels = max_labels
        self.shuffle_labels = shuffle_labels

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

        text = example[self.text_column]
        labels_text = example[self.labels_text_column]
        labels_int = example[self.labels_int_column]

        # randomly shuffle labels if they are more than max_labels
        if isinstance(labels_text, list) and len(labels_text) > self.max_labels:
            combined = list(zip(labels_text, labels_int))
            if self.shuffle_labels:
                random.shuffle(combined)
            labels_text, labels_int = zip(*combined[: self.max_labels])
            labels_int = list(labels_int)  # Convert back to list

        # Tokenize the example
        tokenized: dict[str, torch.Tensor] = self.tokenizer(
            text,
            labels_text,
            return_tensors="pt",
            pad=True,
        )

        # Convert labels to tensor
        if isinstance(labels_int, list):
            labels = torch.tensor(labels_int, dtype=torch.float32)
        else:
            labels = torch.tensor([labels_int], dtype=torch.float32)

        # Return without adding batch dimension (DataLoader will handle batching)
        result = {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "label_mask": tokenized["label_mask"].squeeze(0),
            "labels": labels.reshape(1, -1),  # Ensure labels are 2D
        }

        return result


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # Stack regular tensors
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    label_mask = torch.stack([item["label_mask"] for item in batch])

    # Handle labels which can have different lengths per sample
    labels = [item["labels"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_mask": label_mask,
        "labels": labels,
    }
