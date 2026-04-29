from dataclasses import dataclass, field


@dataclass
class ModelArgs:
    # Model configuration
    model_name: str = field(
        default="microsoft/mdeberta-v3-base",
        metadata={"help": "Pretrained model name or path"},
    )
    model_class: str = field(
        default="DebertaV2PreTrainedModel",
        metadata={
            "help": "Model class to use (e.g., BertPreTrainedModel, DebertaV2PreTrainedModel)"
        },
    )

    # Architecture parameters
    dropout_rate: float = field(
        default=0.1,
        metadata={"help": "Dropout rate for model"},
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
    repulsion_threshold: float = field(
        default=0.3,
        metadata={"help": "Cosine similarity threshold for repulsion penalty"},
    )

    # Data configuration
    dataset_path: str = field(
        default="alexneakameni/synthetic-classification-dataset",
        metadata={"help": "HuggingFace dataset path"},
    )
    dataset_name: str = field(
        default=None,
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
        default=10_000,
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
    max_tokens_per_span: int = field(
        default=64,
        metadata={"help": "Maximum number of tokens per label span"},
    )
    min_text_tokens: int = field(
        default=10,
        metadata={"help": "Minimum number of tokens reserved for text when truncating"},
    )
    min_label_tokens: int = field(
        default=2,
        metadata={"help": "Minimum number of tokens kept per label when truncating"},
    )

    # Training configuration
    early_stopping_patience: int = field(
        default=3,
        metadata={"help": "Early stopping patience"},
    )

    # Whether to augment the training set with additional datasets
    use_additional_datasets: bool = field(
        default=False,
        metadata={
            "help": "Whether to augment the training set with additional datasets defined in training_data.py"
        },
    )
