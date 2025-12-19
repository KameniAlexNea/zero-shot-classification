from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ModelArgs:
    # Model configuration
    model_name: str = field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        metadata={"help": "Pretrained model name or path"},
    )
    model_class: str = field(
        default="BertPreTrainedModel",
        metadata={
            "help": "Model class to use (e.g., BertPreTrainedModel, DebertaV2PreTrainedModel)"
        },
    )

    # Architecture parameters
    projected_dim: Optional[int] = field(
        default=None, metadata={"help": "Projection dimension (None = use hidden_size)"}
    )
    similarity_metric: Literal["dot", "bilinear"] = field(
        default="dot",
        metadata={"help": "Similarity metric: dot, bilinear"},
    )
    dropout_rate: float = field(
        default=0.1,
        metadata={"help": "Dropout rate for model"},
    )
    use_projection_layernorm: bool = field(
        default=True,
        metadata={"help": "Whether to apply LayerNorm after projection"},
    )

    # Loss configuration
    scale_loss: float = field(
        default=10.0,
        metadata={"help": "Multiplier for BCE loss"},
    )
    margin: float = field(
        default=0.1,
        metadata={"help": "Margin for contrastive loss"},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Base temperature for loss scaling"},
    )
    temperature_scale_base: float = field(
        default=10.0,
        metadata={"help": "Base value for temperature scaling"},
    )
    contrastive_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for hard negative mining contrastive loss"},
    )
    separation_loss_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for logit separation regularization"},
    )
    positive_logit_margin: float = field(
        default=1.0,
        metadata={"help": "Minimum desired logit for positive labels"},
    )
    negative_logit_margin: float = field(
        default=-1.0,
        metadata={"help": "Maximum desired logit for negative labels"},
    )
    use_separator_pooling: bool = field(
        default=False,
        metadata={"help": "Use separator token pooling (requires custom [LAB] token)"},
    )

    # Data configuration
    dataset_path: str = field(
        default="alexneakameni/ZSHOT-HARDSET",
        metadata={"help": "HuggingFace dataset path"},
    )
    dataset_name: str = field(
        default="triplet",
        metadata={"help": "Dataset configuration name"},
    )
    max_labels: int = field(
        default=50, metadata={"help": "Maximum number of labels per sample"}
    )
    shuffle_labels: bool = field(
        default=True, metadata={"help": "Shuffle labels (maintains natural proportion)"}
    )
    min_label_length: int = field(
        default=2,
        metadata={"help": "Minimum character length for valid labels"},
    )

    # Tokenizer configuration
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Use fast tokenizer if available"},
    )
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length for the model"},
    )
    lab_cls_token: str = field(
        default="[LAB]",
        metadata={"help": "Separator token for labels ([LAB] or ;)"},
    )

    # Training configuration
    early_stopping_patience: int = field(
        default=3,
        metadata={"help": "Early stopping patience"},
    )
