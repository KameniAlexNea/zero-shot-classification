"""Configuration utilities for GliZNet models."""

from dataclasses import dataclass


class LabelName:
    """Column names for label data."""

    ltext = "ltext"  # Label text column
    lint = "lint"  # Label integer column (0=negative, 1=positive)


@dataclass
class GliZNetDataConfig:
    """Configuration for GliZNet data processing.

    Attributes:
        max_labels: Maximum number of labels per sample (positive labels prioritized)
        shuffle_labels: Whether to shuffle label order
        min_label_length: Minimum character length for valid labels
        text_column: Name of text column in dataset
        positive_column: Name of positive labels column
        negative_column: Name of negative labels column
    """

    max_labels: int = 50
    shuffle_labels: bool = True
    min_label_length: int = 2
    text_column: str = "text"
    positive_column: str = "labels"
    negative_column: str = "not_labels"

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_labels <= 0:
            raise ValueError(f"max_labels must be positive, got {self.max_labels}")

        if self.min_label_length < 0:
            raise ValueError(
                f"min_label_length must be non-negative, got {self.min_label_length}"
            )
