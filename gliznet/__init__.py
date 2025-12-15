"""
GliZNet: Zero-shot text classification system.

A zero-shot classification system inspired by the GLiNER paper, designed to classify text
into positive or negative labels using BERT-based embeddings and contrastive learning.

Architecture:
    - Custom tokenizer that formats input as: [CLS] text [SEP] label1 [LAB] label2 [LAB] ...
    - BERT backbone for encoding text and labels
    - Dual projection heads for text (CLS) and label embeddings
    - Similarity computation (dot product, bilinear, or learned)
    - Multi-component loss (BCE + contrastive + optional Barlow regularization)

Key Features:
    - Two pooling strategies: separator token vs. mean pooling
    - Configurable loss weights and hyperparameters
    - Support for custom separator tokens
    - Efficient batched computation with scatter operations
"""

from .config import (
    GliZNetDataConfig,
    GliZNetTrainingConfig,
    get_default_config,
    get_mean_pooling_config,
    get_separator_pooling_config,
)
from .data import (
    add_tokenized_function,
    collate_fn,
    limit_labels,
    load_dataset,
)
from .model import GliZNetForSequenceClassification, GliZNetOutput
from .tokenizer import GliZNETTokenizer


class LabelName:
    """Column names for label data."""

    ltext = "ltext"  # Label text column
    lint = "lint"  # Label integer column (0=negative, 1=positive)


__all__ = [
    "GliZNetForSequenceClassification",
    "GliZNetOutput",
    "GliZNETTokenizer",
    "GliZNetTrainingConfig",
    "GliZNetDataConfig",
    "get_default_config",
    "get_separator_pooling_config",
    "get_mean_pooling_config",
    "load_dataset",
    "add_tokenized_function",
    "collate_fn",
    "limit_labels",
    "LabelName",
]
