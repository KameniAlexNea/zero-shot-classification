"""Configuration utilities for GliZNet models."""

from dataclasses import dataclass
from typing import Optional


class LabelName:
    """Column names for label data."""

    ltext = "ltext"  # Label text column
    lint = "lint"  # Label integer column (0=negative, 1=positive)


@dataclass
class GliZNetTrainingConfig:
    """Configuration for GliZNet training hyperparameters.

    This class centralizes all training-related hyperparameters to avoid
    scattering magic numbers throughout the codebase.

    Loss Components:
        - BCE loss: Binary cross-entropy between predictions and labels
        - Contrastive loss: Hard negative mining to separate positive/negative labels

    Pooling Strategies:
        - Separator pooling: Uses separator token ([LAB]) embeddings directly
        - Mean pooling: Averages label content token embeddings
    """

    # Model architecture
    projected_dim: Optional[int] = None  # Projection dimension (None = use hidden_size)
    similarity_metric: str = "dot"  # 'dot', 'bilinear', or 'dot_learning'
    dropout_rate: float = 0.1

    # Loss configuration
    scale_loss: float = 10.0  # Multiplier for BCE loss
    margin: float = 0.1  # Margin for contrastive loss (hard negative mining)
    temperature: float = 1.0  # Base temperature for loss scaling
    contrastive_loss_weight: float = 1.0  # Weight for hard negative mining

    # Embedding strategy
    use_separator_pooling: bool = (
        False  # True = use separator tokens, False = mean pooling
    )

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.similarity_metric not in ["dot", "bilinear", "dot"]:
            raise ValueError(
                f"Invalid similarity_metric: {self.similarity_metric}. "
                f"Must be one of: 'dot', 'bilinear', 'dot_learning'"
            )

        if self.projected_dim is not None and self.projected_dim <= 0:
            raise ValueError(
                f"projected_dim must be positive, got {self.projected_dim}"
            )

        if not 0 <= self.dropout_rate < 1:
            raise ValueError(f"dropout_rate must be in [0, 1), got {self.dropout_rate}")

        if self.scale_loss <= 0:
            raise ValueError(f"scale_loss must be positive, got {self.scale_loss}")

        if self.margin < 0:
            raise ValueError(f"margin must be non-negative, got {self.margin}")

    def to_model_kwargs(self) -> dict:
        """Convert to keyword arguments for model initialization."""
        return {
            "projected_dim": self.projected_dim,
            "similarity_metric": self.similarity_metric,
            "dropout_rate": self.dropout_rate,
            "scale_loss": self.scale_loss,
            "margin": self.margin,
            "temperature": self.temperature,
            "contrastive_loss_weight": self.contrastive_loss_weight,
            "use_separator_pooling": self.use_separator_pooling,
        }


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


def get_default_config() -> GliZNetTrainingConfig:
    """Get default training configuration.

    Returns:
        Default GliZNetTrainingConfig with standard hyperparameters
    """
    return GliZNetTrainingConfig()


def get_separator_pooling_config() -> GliZNetTrainingConfig:
    """Get configuration optimized for separator token pooling.

    This configuration is recommended when using custom separator tokens like [LAB].

    Returns:
        GliZNetTrainingConfig with use_separator_pooling=True
    """
    return GliZNetTrainingConfig(
        use_separator_pooling=True,
    )


def get_mean_pooling_config() -> GliZNetTrainingConfig:
    """Get configuration optimized for mean pooling over label tokens.

    This configuration is used when using semicolon (;) as separator.

    Returns:
        GliZNetTrainingConfig with use_separator_pooling=False
    """
    return GliZNetTrainingConfig(
        use_separator_pooling=False,
    )
