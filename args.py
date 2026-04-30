from dataclasses import dataclass, field
from typing import List


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

    supcon_margin: float = field(
        default=0.0,
        metadata={
            "help": "Additive margin for one-vs-negatives loss: negative logits are shifted up "
            "by this value, forcing positives to exceed negatives by at least `m`."
        },
    )
    scoring_method: str = field(
        default="bilinear",
        metadata={"help": "Scoring head: 'bilinear' or 'cosine'"},
    )
    use_cls_label_attention: bool = field(
        default=False,
        metadata={
            "help": "Use CLSLabelAttentionAggregator: single-head self-attention over [CLS, LAB_1..LAB_K] before scoring."
        },
    )
    losses: List[str] = field(
        default_factory=lambda: ["softmax", "repulsion", "bce"],
        metadata={
            "help": "Active loss modules. Any subset of: softmax, repulsion, bce"
        },
    )

    # Data configuration
    eval_size: float = field(
        default=0.05,
        metadata={"help": "Proportion of training data to use for evaluation"},
    )
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
