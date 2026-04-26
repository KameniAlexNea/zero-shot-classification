from typing import Literal, Optional

from transformers import AutoConfig, PretrainedConfig


class GliZNetConfig(PretrainedConfig):
    """Configuration class for GliZNet model.

    Args:
        backbone_model: Name or path of the backbone transformer model
        backbone_config: Backbone model configuration.  Pass explicitly to avoid a
                         network call; resolved lazily at model creation if None.
        projected_dim: Dimension for projection layers (None = use hidden_size, no projection)
        similarity_metric: Similarity computation method ('dot', 'bilinear', or 'cosine')
        dropout_rate: Dropout probability for projections
        use_projection_layernorm: Whether to apply LayerNorm after projection (False = identity when projected_dim=None)
        use_lab_token_for_labels: If True, use [LAB] token embedding as label representation.
                                 If False (default), average all label token embeddings.
                                 Using [LAB] simplifies computation and speeds up inference.
        lab_token_id: Token ID of the [LAB] separator token (set automatically from tokenizer)
        bce_loss_weight: Weight for binary cross-entropy loss
        supcon_loss_weight: Weight for multi-label softmax loss (legacy name; not true SupCon)
        label_repulsion_weight: Weight for label repulsion loss (default 0.0 — disabled).
                               Within-sample repulsion on contextual embeddings is
                               conceptually unsound: semantically related labels on the
                               same input should have similar representations.  Enable
                               only for static/non-contextual label embeddings.
        logit_scale_init: Initial value for learnable temperature scale
        learn_temperature: Whether temperature scale is learnable
        repulsion_threshold: Cosine similarity threshold for repulsion penalty
    """

    model_type = "gliznet"

    def __init__(
        self,
        backbone_model: str = "answerdotai/ModernBERT-base",
        backbone_config: Optional[PretrainedConfig] = None,
        projected_dim: Optional[int] = None,
        similarity_metric: Literal["dot", "bilinear", "cosine"] = "cosine",
        dropout_rate: float = 0.1,
        use_projection_layernorm: bool = False,
        use_lab_token_for_labels: bool = False,
        lab_token_id: Optional[int] = None,
        # Loss weights
        bce_loss_weight: float = 1.0,
        supcon_loss_weight: float = 1.0,
        label_repulsion_weight: float = 0.0,
        # Temperature/scaling
        logit_scale_init: float = 2.0,  # exp(2) ≈ 7.4 for cosine similarity scaling
        learn_temperature: bool = True,
        # Repulsion settings
        repulsion_threshold: float = 0.3,  # Penalize if cosine sim > this
        # Label count upper bound (compile-time constant, eliminates .item() graph breaks)
        max_labels: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.backbone_model = backbone_model
        self.projected_dim = projected_dim
        self.similarity_metric = similarity_metric
        self.dropout_rate = dropout_rate
        self.use_projection_layernorm = use_projection_layernorm
        self.use_lab_token_for_labels = use_lab_token_for_labels
        self.lab_token_id = lab_token_id

        # Loss configuration
        self.bce_loss_weight = bce_loss_weight
        self.supcon_loss_weight = supcon_loss_weight
        self.label_repulsion_weight = label_repulsion_weight
        self.logit_scale_init = logit_scale_init
        self.learn_temperature = learn_temperature
        self.repulsion_threshold = repulsion_threshold
        self.max_labels = max_labels

        # Resolve backbone_config without any network I/O.
        # AutoConfig.from_pretrained() is intentionally NOT called here — config
        # __init__ must be side-effect-free (no network calls, no disk I/O).
        # The model's __init__ resolves it lazily, or callers can pass it explicitly.
        if isinstance(backbone_config, dict):
            backbone_config = AutoConfig.for_model(**backbone_config)
        self.backbone_config = (
            backbone_config  # may be None; resolved at model creation
        )
