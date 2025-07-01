"""
Efficient DataLoader implementation for GliZNet training.

This module provides PyTorch DataLoader-based data loading for improved efficiency
compared to the original HuggingFace datasets approach.
"""

import random
from typing import Dict, List, Union

import datasets
import torch

from . import LabelName
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
            LabelName.ltext: labels,
            LabelName.lint: labels_int,
        }

    ds = datasets.load_dataset(path, name)[split]
    text_column = "text" if "text" in ds.column_names else "sentence"
    ds = ds.map(
        mapper,
    )
    return ds.select_columns(["text", LabelName.ltext, LabelName.lint])


def limit_labels(
    shuffle_labels: bool, labels_text: List[str], labels_int: List[int], max_labels: int
):
    combined = list(zip(labels_text, labels_int))
    if shuffle_labels:
        random.shuffle(combined)
    labels_text, labels_int = zip(*combined[:max_labels])
    return list(labels_text), list(labels_int)


def add_tokenized_function(
    hf_dataset: datasets.Dataset,
    tokenizer: GliZNETTokenizer,
    text_column: str = "text",
    labels_text_column: str = LabelName.ltext,
    labels_int_column: str = LabelName.lint,
    max_labels=50,
    shuffle_labels: bool = True,
) -> datasets.Dataset:
    """
    Tokenizes the HuggingFace dataset using the GliZNETTokenizer.
    """

    def tokenize_function(examples):
        text = examples[text_column]

        # normalize and limit labels in one pass
        raw_texts = examples[labels_text_column]
        raw_ints = examples[labels_int_column]
        # ensure list of lists for unified processing
        if not raw_texts or not isinstance(raw_texts[0], list):
            raw_texts, raw_ints = [raw_texts], [raw_ints]
        processed_texts, processed_labels = [], []
        for lt, li in zip(raw_texts, raw_ints):
            txts, ints = limit_labels(shuffle_labels, lt, li, max_labels)
            processed_texts.append(txts)
            processed_labels.append(torch.tensor(ints, dtype=torch.float32))
        labels_input = (
            processed_texts[0] if len(processed_texts) == 1 else processed_texts
        )
        tokenized: dict[str, torch.Tensor] = tokenizer(text, labels_input)
        labels = processed_labels[0] if len(processed_labels) == 1 else processed_labels

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
