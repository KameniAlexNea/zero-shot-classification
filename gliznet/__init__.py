"""
GliZNet: Zero-shot text classification system.

A zero-shot classification system inspired by the GLiNER paper, designed to classify text
into positive or negative labels using BERT-based embeddings and contrastive learning.

Architecture:
    - Custom tokenizer that formats input as: [CLS] text [SEP] label1 [LAB] label2 [LAB] ...
    - BERT backbone for encoding text and labels
    - Dual projection heads for text (CLS) and label embeddings
    - Similarity computation (dot product, bilinear, or learned)
    - Multi-component loss (BCE + contrastive + optional logit separation)

Key Features:
    - Two pooling strategies: separator token vs. mean pooling
    - Configurable loss weights and hyperparameters
    - Support for custom separator tokens
    - Efficient batched computation with scatter operations
"""

from .config import GliZNetDataConfig, LabelName
from .data import (
    add_tokenized_function,
    collate_fn,
    limit_labels,
    load_dataset,
)
from .model import GliZNetForSequenceClassification, GliZNetOutput
from .predictor import ZeroShotClassificationPipeline
from .tokenizer import GliZNETTokenizer

__all__ = [
    "GliZNetForSequenceClassification",
    "GliZNetOutput",
    "GliZNETTokenizer",
    "ZeroShotClassificationPipeline",
    "GliZNetDataConfig",
    "load_dataset",
    "add_tokenized_function",
    "collate_fn",
    "limit_labels",
    "LabelName",
]
