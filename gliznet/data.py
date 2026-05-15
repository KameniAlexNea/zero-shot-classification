"""
Efficient DataLoader implementation for GliZNet training.

This module provides PyTorch DataLoader-based data loading for improved efficiency
compared to the original HuggingFace datasets approach.
"""

import os
import random
from typing import Dict, List, Optional

import datasets
import torch
from torch.nn.utils.rnn import pad_sequence

from .augmentation import AugmentationPipeline, LabelAugmentationPipeline, LabelLimit
from .tokenizer import GliZNETTokenizer
from .training_config import LabelName


def _add_cross_sample_negatives(
    batch_label_texts: list[list[str]],
    batch_label_ints: list[list[int]],
    prob: float = 0.15,
    max_added: int = 2,
) -> tuple[list[list[str]], list[list[int]]]:
    """Add positive labels from other samples as hard negatives.

    For each sample (with probability ``prob``), finds positive labels from
    other samples that share tokens with this sample's positives (semantically
    close but different labels). Adds them as negatives.

    Args:
        batch_label_texts: List of label text lists per sample.
        batch_label_ints: List of label int lists per sample.
        prob: Probability of applying to each sample.
        max_added: Maximum negatives to add per sample.

    Returns:
        Modified (batch_label_texts, batch_label_ints).
    """
    if len(batch_label_texts) < 2:
        return batch_label_texts, batch_label_ints

    # Collect positive labels and their tokens per sample
    sample_positives: list[list[str]] = []
    sample_pos_tokens: list[set[str]] = []
    for texts, ints in zip(batch_label_texts, batch_label_ints):
        pos = [t for t, i in zip(texts, ints) if i == 1]
        tokens = set()
        for p in pos:
            tokens.update(p.lower().replace("_", " ").split())
        sample_positives.append(pos)
        sample_pos_tokens.append(tokens)

    for i in range(len(batch_label_texts)):
        if random.random() >= prob:
            continue

        existing = set(batch_label_texts[i])
        my_tokens = sample_pos_tokens[i]
        if not my_tokens:
            continue

        candidates = []
        for j in range(len(batch_label_texts)):
            if j == i:
                continue
            for pos_label in sample_positives[j]:
                if pos_label in existing:
                    continue
                label_tokens = set(pos_label.lower().replace("_", " ").split())
                overlap = len(label_tokens & my_tokens)
                if overlap > 0:
                    candidates.append((overlap, pos_label))

        if not candidates:
            continue

        # Sort by overlap descending, take top-k
        candidates.sort(key=lambda x: x[0], reverse=True)
        added = 0
        for _, label in candidates:
            if label not in existing:
                batch_label_texts[i] = batch_label_texts[i] + [label]
                batch_label_ints[i] = batch_label_ints[i] + [0]
                existing.add(label)
                added += 1
                if added >= max_added:
                    break

    return batch_label_texts, batch_label_ints


def load_dataset(
    path: str,
    name: str = None,
    split: str = "train",
    text_column: str = "text",
    positive_column: str = "labels",
    negative_column: str = "not_labels",
    shuffle_labels: bool = True,
    min_label_length: int = 2,
):
    """Load and preprocess a HuggingFace dataset for GliZNet training.

    The dataset must have columns for text, positive labels, and negative labels.
    The function normalises it into the GliZNet format:
        ``text``, ``ltext`` (list of label strings), ``lint`` (list of 0/1 ints).

    Args:
        path: HuggingFace dataset path (e.g., 'user/dataset-name')
        name: Dataset configuration name (optional)
        split: Dataset split to load ('train', 'validation', 'test')
        text_column: Column name containing text samples
        positive_column: Column name containing positive labels
        negative_column: Column name containing negative labels
        shuffle_labels: Whether to shuffle labels within each sample
        min_label_length: Minimum character length for valid labels

    Returns:
        HuggingFace Dataset with columns: 'text', LabelName.ltext, LabelName.lint
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
    ds = ds.map(mapper)
    ds = ds.filter(lambda x: len(x[LabelName.ltext]) > 0)

    return ds.select_columns(["text", LabelName.ltext, LabelName.lint])


def add_tokenized_function(
    hf_dataset: datasets.Dataset,
    tokenizer: GliZNETTokenizer,
    text_column: str = "text",
    labels_text_column: str = LabelName.ltext,
    labels_int_column: str = LabelName.lint,
    max_labels=50,
    shuffle_labels: bool = True,
    as_transform: bool = True,
    augmentation_pipeline: Optional[AugmentationPipeline] = None,
    label_augmentation_pipeline: Optional[LabelAugmentationPipeline] = None,
    cross_sample_neg_prob: float = 0.0,
    cross_sample_neg_max: int = 2,
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
        augmentation_pipeline: Optional AugmentationPipeline to apply to text (training only)
        label_augmentation_pipeline: Optional LabelAugmentationPipeline to apply to labels

    Returns:
        Tokenized dataset
    """
    # Default label augmentation: just LabelLimit (same behavior as before)
    if label_augmentation_pipeline is None:
        label_augmentation_pipeline = LabelAugmentationPipeline(
            [LabelLimit(max_labels=max_labels, shuffle_labels=shuffle_labels)]
        )

    def tokenize_function(examples):
        # Handle batched input format
        texts = examples[text_column]
        raw_texts_batch = examples[labels_text_column]
        raw_ints_batch = examples[labels_int_column]

        # Prepare (text, labels) tuples for tokenizer
        tokenizer_inputs = []
        all_label_texts = []
        all_label_ints = []

        for text, raw_texts, raw_ints in zip(texts, raw_texts_batch, raw_ints_batch):
            # Apply text augmentation pipeline if provided
            if augmentation_pipeline is not None:
                text = augmentation_pipeline(text)

            # Apply label augmentation pipeline
            label_texts, label_ints = label_augmentation_pipeline(raw_texts, raw_ints)

            tokenizer_inputs.append((text, label_texts))
            all_label_texts.append(label_texts)
            all_label_ints.append(label_ints)

        # Cross-sample hard negatives: steal positives from other samples
        if cross_sample_neg_prob > 0:
            all_label_texts, all_label_ints = _add_cross_sample_negatives(
                all_label_texts, all_label_ints,
                prob=cross_sample_neg_prob,
                max_added=cross_sample_neg_max,
            )
            # Rebuild tokenizer_inputs with updated labels
            tokenizer_inputs = [
                (ti[0], lt) for ti, lt in zip(tokenizer_inputs, all_label_texts)
            ]

        labels_batch = [
            torch.tensor(ints, dtype=torch.float32) for ints in all_label_ints
        ]

        # Tokenize all examples in batch
        tokenized: dict[str, torch.Tensor] = tokenizer(
            tokenizer_inputs, return_tensors="pt"
        )

        # Determine how many labels actually fit by checking lmask
        # lmask contains label IDs (1, 2, 3, ...) for each label's tokens
        truncated_labels = []

        for lmask_row, label_tensor in zip(tokenized["lmask"], labels_batch):
            num_fitted = int(lmask_row.max().item()) if lmask_row.any() else 0
            truncated_labels.append(label_tensor[:num_fitted])

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "lmask": tokenized["lmask"],
            "labels": truncated_labels,
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
        }

    # Dynamic padding: pad input_ids / attention_mask / lmask to the longest
    # sequence in the batch. This avoids wasting compute on padding tokens.
    input_ids = pad_sequence(
        [item["input_ids"] for item in batch], batch_first=True, padding_value=0
    )
    attention_mask = pad_sequence(
        [item["attention_mask"] for item in batch], batch_first=True, padding_value=0
    )
    lmask = pad_sequence(
        [item["lmask"] for item in batch], batch_first=True, padding_value=0
    )

    # Pad labels (variable length per sample) here at collation time
    labels = pad_sequence(
        [item["labels"] for item in batch], batch_first=True, padding_value=-100
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "lmask": lmask,
        "labels": labels,
    }
