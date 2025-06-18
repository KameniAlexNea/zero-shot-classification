"""
Efficient DataLoader implementation for GliZNet training.

This module provides PyTorch DataLoader-based data loading for improved efficiency
compared to the original HuggingFace datasets approach.
"""

import random
from typing import Dict, List, Union

import datasets
import torch

from .tokenizer import GliZNETTokenizer


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


def limit_labels(
    shuffle_labels: bool, labels_text: List[str], labels_int: List[int], max_labels: int
):
    combined = list(zip(labels_text, labels_int))
    if shuffle_labels:
        random.shuffle(combined)
    labels_text, labels_int = zip(*combined[:max_labels])
    labels_int = list(labels_int)  # Convert back to list
    return labels_text, labels_int


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
        if labels_text and isinstance(labels_text[0], list):
            all_labels = [
                limit_labels(shuffle_labels, ltext, lint, max_labels)
                for ltext, lint in zip(labels_text, labels_int)
            ]
            labels_text, labels_int = zip(*all_labels)
            labels_text = [list(i) for i in labels_text]
            labels_int = [list(i) for i in labels_int]
        else:
            labels_text, labels_int = limit_labels(
                shuffle_labels, labels_text, labels_int, max_labels
            )
            labels_text = list(labels_text)

        # Tokenize the example
        tokenized: dict[str, torch.Tensor] = tokenizer(
            text,
            labels_text,
            return_tensors="pt",
            pad=True,
        )

        # Convert labels to tensor
        if isinstance(labels_int[0], list):
            labels = [torch.tensor([lint], dtype=torch.float32) for lint in labels_int]
        else:
            labels = torch.tensor([labels_int], dtype=torch.float32)

        # Return without adding batch dimension (DataLoader will handle batching)
        result: dict[str, Union[torch.Tensor, list[torch.Tensor]]] = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "lmask": tokenized["lmask"],
            "labels": labels,
        }
        if len(tokenized["input_ids"].shape) == 1:
            # If the input is a single string, we need to add a batch dimension
            result["input_ids"] = result["input_ids"].unsqueeze(0)
            result["attention_mask"] = result["attention_mask"].unsqueeze(0)
            result["lmask"] = result["lmask"].unsqueeze(0)

        return result

    return hf_dataset.with_transform(
        tokenize_function,
    )


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # Stack regular tensors
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    lmask = torch.stack([item["lmask"] for item in batch])

    # Handle labels which can have different lengths per sample
    labels = [item["labels"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "lmask": lmask,
        "labels": labels,
    }
