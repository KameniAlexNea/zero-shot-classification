"""Augmentation package for GliZNet training.

Submodules:
    - text: character-level and word-level text augmentations
    - label: label ratio/scenario augmentations
"""

from pathlib import Path

import yaml

from .label import (
    LABEL_AUGMENTATION_REGISTRY,
    LabelAugmentation,
    LabelAugmentationPipeline,
    LabelLimit,
    LabelSimplification,
    ScenarioAwareSampler,
)
from .text import (
    AUGMENTATION_REGISTRY,
    AugmentationPipeline,
    RandomCaseChange,
    SuffixTruncation,
    TextAugmentation,
)

__all__ = [
    # text
    "TextAugmentation",
    "SuffixTruncation",
    "RandomCaseChange",
    "AugmentationPipeline",
    "AUGMENTATION_REGISTRY",
    # label
    "LabelAugmentation",
    "LabelLimit",
    "LabelSimplification",
    "ScenarioAwareSampler",
    "LabelAugmentationPipeline",
    "LABEL_AUGMENTATION_REGISTRY",
    # loaders
    "load_augmentation_pipeline",
    "load_label_augmentation_pipeline",
]


def load_augmentation_pipeline(config_path: str) -> AugmentationPipeline:
    """Load a text augmentation pipeline from a YAML config file.

    Args:
        config_path: Path to YAML config.

    Returns:
        Configured AugmentationPipeline
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Augmentation config not found: {config_path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    entries = config.get("augmentations", [])
    augmentations = []
    for entry in entries:
        name = entry["name"]
        prob = entry.get("prob", 0.3)
        params = entry.get("params", {})

        if name not in AUGMENTATION_REGISTRY:
            raise ValueError(
                f"Unknown augmentation '{name}'. "
                f"Available: {list(AUGMENTATION_REGISTRY.keys())}"
            )

        factory = AUGMENTATION_REGISTRY[name]
        augmentations.append((prob, factory(**params)))

    return AugmentationPipeline(augmentations)


def load_label_augmentation_pipeline(
    config_path: str,
    max_labels: int | None = None,
) -> LabelAugmentationPipeline:
    """Load a label augmentation pipeline from a YAML config file.

    Args:
        config_path: Path to YAML config.
        max_labels: If provided, overrides max_labels in LabelLimit config
            (useful for passing the training args value).

    Returns:
        Configured LabelAugmentationPipeline
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Augmentation config not found: {config_path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    entries = config.get("label_augmentations", [])
    if not entries:
        raise ValueError(
            f"No 'label_augmentations' section in {config_path}. "
            "Define label augmentations in the config file."
        )

    augmentations = []
    for entry in entries:
        name = entry["name"]
        params = entry.get("params", {})

        if name not in LABEL_AUGMENTATION_REGISTRY:
            raise ValueError(
                f"Unknown label augmentation '{name}'. "
                f"Available: {list(LABEL_AUGMENTATION_REGISTRY.keys())}"
            )

        # Override max_labels from args if provided
        if name == "LabelLimit" and max_labels is not None:
            params["max_labels"] = max_labels

        aug_cls = LABEL_AUGMENTATION_REGISTRY[name]
        augmentations.append(aug_cls(**params))

    return LabelAugmentationPipeline(augmentations)
