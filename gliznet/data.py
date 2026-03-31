"""
Efficient DataLoader implementation for GliZNet training.

This module provides PyTorch DataLoader-based data loading for improved efficiency
compared to the original HuggingFace datasets approach.
"""

import os
import random
from typing import Dict, List

import datasets
import torch
from torch.nn.utils.rnn import pad_sequence

from .tokenizer import GliZNETTokenizer
from .training_config import LabelName


def sample_dataset(ds: datasets.Dataset, max_size: int = 50_000):
    if len(ds) < max_size:
        return ds
    index = list(range(len(ds)))
    rand = random.Random(42)
    rand.shuffle(index)
    return ds.select(index[:max_size])


def load_dataset(
    path: str = "alexneakameni/ZSHOT-HARDSET",
    name: str = "triplet",
    split: str = "train",
    text_column: str = "sentence",
    positive_column: str = "labels",
    negative_column: str = "not_labels",
    shuffle_labels: bool = True,
    min_label_length: int = 2,
):
    """Load and preprocess a HuggingFace dataset for GliZNet training.

    Args:
        path: HuggingFace dataset path (e.g., 'user/dataset-name')
        name: Dataset configuration name
        split: Dataset split to load ('train', 'validation', 'test')
        text_column: Column name containing text samples
        positive_column: Column name containing positive labels
        negative_column: Column name containing negative labels
        shuffle_labels: Whether to shuffle labels within each sample
        min_label_length: Minimum character length for valid labels

    Returns:
        HuggingFace Dataset with columns: 'text', 'labels_text', 'labels_int'
    """

    def mapper(x: dict[str, list[str]]):
        pos = [
            i.strip() for i in x[positive_column] if len(i.strip()) > min_label_length
        ]
        neg = [
            i.strip() for i in x[negative_column] if len(i.strip()) > min_label_length
        ]
        labels = pos + neg
        labels_int = [1] * len(pos) + [0] * len(neg)

        # Handle shuffling
        if shuffle_labels and labels:
            combined = list(zip(labels, labels_int))
            random.shuffle(combined)
            labels, labels_int = zip(*combined)
            labels = list(labels)
            labels_int = list(labels_int)

        return {
            "text": x[text_column],
            LabelName.ltext: labels,
            LabelName.lint: labels_int,
        }

    ds = datasets.load_dataset(path, name)[split]
    if split == "train":
        arxiv_ds = datasets.load_from_disk("arxiv_synthetic_data/based_dataset")
        arxiv_ds = sample_dataset(arxiv_ds, max_size=5_000)
        ds: datasets.Dataset = datasets.concatenate_datasets([ds, arxiv_ds])
        ds = ds.shuffle(seed=42)
    text_column = "text" if "text" in ds.column_names else "sentence"
    ds = ds.map(mapper)

    # Filter out samples with no labels after filtering by min_label_length
    ds = ds.filter(lambda x: len(x[LabelName.ltext]) > 0)

    return ds.select_columns(["text", LabelName.ltext, LabelName.lint])


def limit_labels(
    labels_text: List[str],
    labels_int: List[int],
    shuffle_labels: bool,
    max_labels: int,
    remove_underscores: float = 0.9,
):
    """Limit the number of labels while maintaining natural proportion.

    Args:
        labels_text: List of label text strings
        labels_int: List of label integers (1 for positive, 0 for negative)
        shuffle_labels: Whether to shuffle labels randomly
        max_labels: Maximum number of labels to keep

    Returns:
        Tuple of (limited_labels_text, limited_labels_int)

    Strategy:
        1. Randomly shuffle all labels if requested
        2. Take first max_labels samples
        3. This naturally maintains the original proportion of positive/negative labels
    """
    labels_text = [
        i.replace("_", " ") if random.random() < remove_underscores else i
        for i in labels_text
    ]

    # Combine labels into pairs
    combined = list(zip(labels_text, labels_int))

    if shuffle_labels and combined:
        random.shuffle(combined)
        # Randomly select between 1 and max_labels
        num_labels = random.randint(1, min(max_labels, len(combined)))
        selected_pairs = combined[:num_labels]
    else:
        selected_pairs = combined[:max_labels]

    if not selected_pairs:
        return [], []

    labels_text, labels_int = zip(*selected_pairs)
    return list(labels_text), list(labels_int)


def add_tokenized_function(
    hf_dataset: datasets.Dataset,
    tokenizer: GliZNETTokenizer,
    text_column: str = "text",
    labels_text_column: str = LabelName.ltext,
    labels_int_column: str = LabelName.lint,
    max_labels=50,
    shuffle_labels: bool = True,
    as_transform: bool = True,
) -> datasets.Dataset:
    """Tokenize the HuggingFace dataset using the GliZNETTokenizer.

    Args:
        hf_dataset: Input HuggingFace dataset
        tokenizer: GliZNETTokenizer instance
        text_column: Column name containing text
        labels_text_column: Column name containing label texts
        labels_int_column: Column name containing label integers (0/1)
        max_labels: Maximum number of labels to keep per sample
        shuffle_labels: Whether to shuffle labels (positives are always preserved)
        as_transform: If True, apply as lazy transform; if False, map eagerly

    Returns:
        Tokenized dataset
    """

    def tokenize_function(examples):
        # Handle batched input format
        texts = examples[text_column]
        raw_texts_batch = examples[labels_text_column]
        raw_ints_batch = examples[labels_int_column]

        # Prepare (text, labels) tuples for tokenizer
        tokenizer_inputs = []
        labels_batch = []

        for text, raw_texts, raw_ints in zip(texts, raw_texts_batch, raw_ints_batch):
            # Process labels for this example
            label_texts, label_ints = limit_labels(
                raw_texts, raw_ints, shuffle_labels, max_labels
            )

            tokenizer_inputs.append((text, label_texts))
            labels_batch.append(torch.tensor(label_ints, dtype=torch.float32))

        # Tokenize all examples in batch
        tokenized: dict[str, torch.Tensor] = tokenizer(
            tokenizer_inputs, return_tensors="pt"
        )

        # Determine how many labels actually fit by checking lmask
        # lmask contains label IDs (1, 2, 3, ...) for each label's tokens
        truncated_labels = []
        num_labels_per_text = []

        for lmask_row, label_tensor in zip(tokenized["lmask"], labels_batch):
            # Count unique non-zero label IDs to see how many labels fit
            num_fitted = int(lmask_row.max().item()) if lmask_row.any() else 0
            truncated_labels.append(label_tensor[:num_fitted])
            num_labels_per_text.append(num_fitted)

        # Pad labels to same length
        labels_padded = pad_sequence(
            truncated_labels, batch_first=True, padding_value=-100
        )
        num_labels_tensor = torch.tensor(num_labels_per_text, dtype=torch.long)

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "lmask": tokenized["lmask"],
            "labels": labels_padded,
            "num_labels": num_labels_tensor,
        }

    if as_transform:
        return hf_dataset.with_transform(tokenize_function)

    return hf_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=10_000,
        remove_columns=hf_dataset.column_names,
        desc="Tokenizing dataset",
        num_proc=os.cpu_count(),
    )


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for batching samples.

    Args:
        batch: List of tokenized samples

    Returns:
        Batched tensors with proper padding
    """
    # Handle single dict (from with_transform)
    if isinstance(batch, dict):
        return {
            "input_ids": (
                batch["input_ids"]
                if isinstance(batch["input_ids"], torch.Tensor)
                else torch.tensor(batch["input_ids"])
            ),
            "attention_mask": (
                batch["attention_mask"]
                if isinstance(batch["attention_mask"], torch.Tensor)
                else torch.tensor(batch["attention_mask"])
            ),
            "lmask": (
                batch["lmask"]
                if isinstance(batch["lmask"], torch.Tensor)
                else torch.tensor(batch["lmask"])
            ),
            "labels": (
                batch["labels"]
                if isinstance(batch["labels"], torch.Tensor)
                else torch.tensor(batch["labels"])
            ),
            "num_labels": (
                batch["num_labels"]
                if isinstance(batch["num_labels"], torch.Tensor)
                else torch.tensor(batch["num_labels"])
            ),
        }

    # Stack regular tensors (already tensors from tokenizer)
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    lmask = torch.stack([item["lmask"] for item in batch])
    num_labels = torch.stack([item["num_labels"] for item in batch])

    # Pad labels (variable length per sample) - already padded but may need re-padding
    labels = pad_sequence(
        [item["labels"] for item in batch], batch_first=True, padding_value=-100
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "lmask": lmask,
        "labels": labels,
        "num_labels": num_labels,
    }
