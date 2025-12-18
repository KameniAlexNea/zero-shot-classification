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

from .config import LabelName
from .tokenizer import GliZNETTokenizer


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
    text_column = "text" if "text" in ds.column_names else "sentence"
    ds = ds.map(mapper)

    # Filter out samples with no labels after filtering by min_label_length
    ds = ds.filter(lambda x: len(x[LabelName.ltext]) > 0)

    return ds.select_columns(["text", LabelName.ltext, LabelName.lint])


def limit_labels(
    labels_text: List[str], labels_int: List[int], shuffle_labels: bool, max_labels: int
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
    labels_text = [i.replace("_", " ") for i in labels_text]

    # Combine labels into pairs
    combined = list(zip(labels_text, labels_int))

    # Shuffle randomly if requested (maintains natural proportion)
    if shuffle_labels:
        random.shuffle(combined)

    # Take up to max_labels
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

    Note:
        Token dropout should be applied dynamically in collate_fn, not here,
        to ensure different dropout masks across epochs.
    """

    def tokenize_function(examples):
        # Handle batched input format: text=['hello'], ltext=[[...]], lint=[[...]]
        texts = examples[text_column]
        raw_texts_batch = examples[labels_text_column]
        raw_ints_batch = examples[labels_int_column]

        # First step: prepare all labels data
        processed_lints_batch = []
        processed_ltexts_batch = []

        for raw_texts, raw_ints in zip(raw_texts_batch, raw_ints_batch):
            # Process labels for this example
            txts, ints = limit_labels(raw_texts, raw_ints, shuffle_labels, max_labels)
            processed_ltexts_batch.append(txts)
            processed_lints_batch.append(torch.tensor(ints, dtype=torch.float32))

        # Second step: batch tokenize everything at once (without token dropout)
        tokenized = tokenizer(texts, processed_ltexts_batch, token_dropout=0.0)

        # Third step: truncate labels to match what actually fit after tokenization
        truncated_labels = []
        for label_tensor, num_fitted in zip(
            processed_lints_batch, tokenized["num_labels_fitted"]
        ):
            # Only keep labels that actually fit in the tokenized sequence
            truncated_labels.append(label_tensor[:num_fitted])

        # Return the results (without num_labels_fitted in the dataset)
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "lmask": tokenized["lmask"],
            "labels": truncated_labels,
        }

    if as_transform:
        return hf_dataset.with_transform(
            tokenize_function,
        )
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
        batch: List of tokenized samples or single dict (from with_transform)

    Returns:
        Batched tensors with proper padding

    Note:
        If you need token dropout, apply it here by masking input_ids randomly.
        This ensures different dropout masks across epochs.
    """
    if isinstance(batch, dict):
        return {
            "input_ids": torch.tensor(batch["input_ids"]),
            "attention_mask": torch.tensor(batch["attention_mask"]),
            "lmask": torch.tensor(batch["lmask"]),
            "labels": pad_sequence(
                [torch.tensor(lab) for lab in batch["labels"]],
                batch_first=True,
                padding_value=-100,
            ),
        }
    # Stack regular tensors
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    lmask = torch.stack([item["lmask"] for item in batch])

    # Handle labels which can have different lengths per sample
    labels = pad_sequence(
        [item["labels"] for item in batch], batch_first=True, padding_value=-100
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "lmask": lmask,
        "labels": labels,
    }
