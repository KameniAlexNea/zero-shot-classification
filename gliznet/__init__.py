"""
GliZNet: Zero-shot text classification system.

A zero-shot classification system inspired by the GLiNER paper, designed to classify text
into positive or negative labels using BERT-based embeddings and contrastive learning.

Architecture:
    - Custom tokenizer that formats input as: [CLS] text [SEP] [LAB] label1 [LAB] label2 ...
    - Backbone transformer for encoding text and labels jointly
    - Token-level cross-attention to build label-specific text representations
    - Bilinear head for scoring text-label pairs
    - Multi-component loss (BCE + multi-label softmax + optional label repulsion)
"""

from .augmentation import (
    AugmentationPipeline,
    LabelAugmentationPipeline,
    LabelLimit,
    RatioEnforcement,
    RatioEnforcementSelector,
    ScenarioAwareSampler,
    load_augmentation_pipeline,
    load_label_augmentation_pipeline,
)
from .data import (
    add_tokenized_function,
    collate_fn,
    load_dataset,
)
from .model import GliZNetForSequenceClassification, GliZNetOutput
from .predictor import ZeroShotClassificationPipeline
from .tokenizer import GliZNETTokenizer
from .training_config import GliZNetDataConfig, LabelName

__all__ = [
    "GliZNetForSequenceClassification",
    "GliZNetOutput",
    "GliZNETTokenizer",
    "ZeroShotClassificationPipeline",
    "GliZNetDataConfig",
    "load_dataset",
    "add_tokenized_function",
    "collate_fn",
    "LabelLimit",
    "RatioEnforcement",
    "RatioEnforcementSelector",
    "LabelAugmentationPipeline",
    "AugmentationPipeline",
    "load_augmentation_pipeline",
    "load_label_augmentation_pipeline",
    "LabelName",
]
