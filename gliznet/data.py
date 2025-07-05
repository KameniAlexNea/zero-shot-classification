"""
Efficient DataLoader implementation for GliZNet training.

This module provides PyTorch DataLoader-based data loading for improved efficiency
compared to the original HuggingFace datasets approach.
"""

import random
from typing import Dict, List

import datasets
import torch
from torch.nn.utils.rnn import pad_sequence

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
    min_label_length: int = 2,
):
    def mapper(x: dict[str, list[str]]):
        pos = [
            i.strip() for i in x[positive_column] if len(i.strip()) > min_label_length
        ]
        neg = [
            i.strip() for i in x[negative_column] if len(i.strip()) > min_label_length
        ]
        labels = pos + neg
        labels_int = [1] * len(pos) + [0] * len(neg)
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
    labels_text: List[str], labels_int: List[int], shuffle_labels: bool, max_labels: int
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

        # Second step: batch tokenize everything at once
        tokenized = tokenizer(texts, processed_ltexts_batch)

        # Return the results
        return {
            **tokenized,
            "labels": processed_lints_batch,
        }

    return hf_dataset.with_transform(
        tokenize_function,
    )


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
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
