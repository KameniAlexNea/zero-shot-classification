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
    similarity_metric: Literal["dot", "bilinear", "cosine"] = field(
        default="cosine",
        metadata={"help": "Similarity metric: dot, bilinear, cosine"},
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
    bce_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for binary cross-entropy loss"},
    )
    supcon_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for supervised contrastive loss"},
    )
    label_repulsion_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for label repulsion loss (prevents embedding collapse)"
        },
    )
    logit_scale_init: float = field(
        default=2.0,
        metadata={"help": "Initial value for learnable logit scale (exp(2) ≈ 7.4)"},
    )
    learn_temperature: bool = field(
        default=True,
        metadata={"help": "Whether to learn temperature/scale parameter"},
    )
    repulsion_threshold: float = field(
        default=0.3,
        metadata={"help": "Cosine similarity threshold for repulsion penalty"},
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
    max_extended_ds_size: int = field(
        default=50_000,
        metadata={"help": "Max size of the extended dataset added for training"},
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
