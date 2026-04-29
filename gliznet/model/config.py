from typing import Optional

from transformers import AutoConfig, PretrainedConfig


class GliZNetConfig(PretrainedConfig):
    """Configuration class for GliZNet model.

    Args:
        backbone_model: Name or path of the backbone transformer model
        backbone_config: Backbone model configuration.  Pass explicitly to avoid a
                         network call; resolved lazily at model creation if None.
        dropout_rate: Dropout probability for projections
        lab_token_id: Token ID of the [LAB] separator token (set automatically from tokenizer)
        bce_loss_weight: Weight for binary cross-entropy loss
        supcon_loss_weight: Weight for multi-label softmax loss (legacy name; not true SupCon)
        label_repulsion_weight: Weight for label repulsion loss (default 0.0 — disabled).
        repulsion_threshold: Cosine similarity threshold for repulsion penalty
    """

    model_type = "gliznet"

    def __init__(
        self,
        backbone_model: str = "answerdotai/ModernBERT-base",
        backbone_config: Optional[PretrainedConfig] = None,
        dropout_rate: float = 0.1,
        lab_token_id: Optional[int] = None,
        # Loss weights
        bce_loss_weight: float = 1.0,
        supcon_loss_weight: float = 1.0,
        label_repulsion_weight: float = 0.0,
        # One-vs-negatives margin: negatives are shifted up by this value before logsumexp,
        # forcing the model to maintain a gap of at least `m` between positive and negative logits.
        supcon_margin: float = 0.0,
        # Repulsion settings
        repulsion_threshold: float = 0.3,
        # Label count upper bound (compile-time constant, eliminates .item() graph breaks)
        max_labels: int = 20,
        # Scoring head: "bilinear" or "cosine"
        scoring_method: str = "bilinear",
        # Active loss modules — any subset of LOSS_REGISTRY keys
        losses: list = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if max_labels <= 0:
            raise ValueError("max_labels must be a positive integer")

        self.backbone_model = backbone_model
        self.dropout_rate = dropout_rate
        self.lab_token_id = lab_token_id

        # Loss configuration
        self.bce_loss_weight = bce_loss_weight
        self.supcon_loss_weight = supcon_loss_weight
        self.label_repulsion_weight = label_repulsion_weight
        self.supcon_margin = supcon_margin
        self.repulsion_threshold = repulsion_threshold
        self.max_labels = max_labels
        self.scoring_method = scoring_method
        self.losses = list(losses) if losses is not None else ["softmax", "repulsion", "bce"]

        # Resolve backbone_config without any network I/O.
        # AutoConfig.from_pretrained() is intentionally NOT called here — config
        # __init__ must be side-effect-free (no network calls, no disk I/O).
        # The model's __init__ resolves it lazily, or callers can pass it explicitly.
        if isinstance(backbone_config, dict):
            backbone_config = AutoConfig.for_model(**backbone_config)
        self.backbone_config = (
            backbone_config  # may be None; resolved at model creation
        )
