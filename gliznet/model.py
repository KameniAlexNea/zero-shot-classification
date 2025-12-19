from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModel,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import ModelOutput

if TYPE_CHECKING:
    from gliznet.tokenizer import GliZNETTokenizer


# ============================================================================
# Configuration
# ============================================================================


class GliZNetConfig(PretrainedConfig):
    """Configuration class for GliZNet model.

    Args:
        backbone_model: Name or path of the backbone transformer model
        backbone_config: Backbone model configuration (loaded automatically if None)
        projected_dim: Dimension for projection layers (None = use hidden_size)
        similarity_metric: Similarity computation method ('dot' or 'bilinear')
        dropout_rate: Dropout probability for projections
        use_projection_layernorm: Whether to apply LayerNorm after projection
        scale_loss: Multiplier for BCE loss
        margin: Margin for contrastive loss
        contrastive_loss_weight: Weight for contrastive loss
        temperature: Temperature for contrastive loss scaling
        temperature_scale_base: Base value for temperature scaling
        separation_loss_weight: Weight for logit separation regularization
        positive_logit_margin: Minimum desired logit for positive labels
        negative_logit_margin: Maximum desired logit for negative labels
    """

    model_type = "gliznet"

    def __init__(
        self,
        backbone_model: str = "microsoft/deberta-v3-small",
        backbone_config: Optional[dict] = None,
        projected_dim: Optional[int] = None,
        similarity_metric: Literal["dot", "bilinear"] = "dot",
        dropout_rate: float = 0.1,
        use_projection_layernorm: bool = True,
        scale_loss: float = 10.0,
        margin: float = 0.1,
        contrastive_loss_weight: float = 1.0,
        temperature: float = 1.0,
        temperature_scale_base: float = 10.0,
        separation_loss_weight: float = 0.1,
        positive_logit_margin: float = 1.0,
        negative_logit_margin: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.backbone_model = backbone_model
        self.projected_dim = projected_dim
        self.similarity_metric = similarity_metric
        self.dropout_rate = dropout_rate
        self.use_projection_layernorm = use_projection_layernorm

        # Loss configuration
        self.scale_loss = scale_loss
        self.margin = margin
        self.contrastive_loss_weight = contrastive_loss_weight
        self.temperature = temperature
        self.temperature_scale_base = temperature_scale_base
        self.separation_loss_weight = separation_loss_weight
        self.positive_logit_margin = positive_logit_margin
        self.negative_logit_margin = negative_logit_margin

        # Load and store backbone config
        if backbone_config is None:
            backbone_config = AutoConfig.from_pretrained(backbone_model).to_dict()
        self.backbone_config = backbone_config


# ============================================================================
# Model Outputs
# ============================================================================


@dataclass
class GliZNetOutput(ModelOutput):
    """Output class for GliZNet model.

    Args:
        loss: Training loss (optional)
        logits: Classification logits
        batch_indices: Batch index for each prediction
        label_ids: Label ID for each prediction
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.Tensor] = None
    batch_indices: Optional[torch.Tensor] = None
    label_ids: Optional[torch.Tensor] = None


# ============================================================================
# Model Components
# ============================================================================


class SimilarityHead(nn.Module):
    """Computes similarity between text and label representations."""

    def __init__(self, config: GliZNetConfig, projected_dim: int):
        super().__init__()
        self.config = config

        if config.similarity_metric == "bilinear":
            self.classifier = nn.Bilinear(projected_dim, projected_dim, 1)
        elif config.similarity_metric == "dot":
            self.classifier = nn.Linear(projected_dim, 1)
        else:
            raise ValueError(
                f"Unknown similarity_metric: {config.similarity_metric}. "
                "Choose 'dot' or 'bilinear'."
            )

    def forward(
        self, text_repr: torch.Tensor, label_repr: torch.Tensor
    ) -> torch.Tensor:
        """Compute similarity scores.

        Args:
            text_repr: Text representations (N, D)
            label_repr: Label representations (N, D)

        Returns:
            Similarity scores (N, 1)
        """
        if self.config.similarity_metric == "bilinear":
            return self.classifier(text_repr, label_repr)
        return self.classifier(text_repr * label_repr)


class LabelAggregator(nn.Module):
    """Aggregates label token embeddings and computes similarities."""

    def __init__(
        self,
        config: GliZNetConfig,
        text_projector: nn.Module,
        label_projector: nn.Module,
        similarity_head: nn.Module,
    ):
        super().__init__()
        self.config = config
        self.text_projector = text_projector
        self.label_projector = label_projector
        self.similarity_head = similarity_head
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self, hidden_states: torch.Tensor, lmask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aggregate label representations and compute similarities.

        Args:
            hidden_states: Encoder outputs (B, L, H)
            lmask: Label mask where >0 indicates label tokens (B, L)

        Returns:
            logits: Similarity scores (N, 1)
            batch_indices: Batch index for each score (N,)
            label_ids: Label ID for each score (N,)
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # Project CLS tokens
        cls_repr = self.dropout(self.text_projector(hidden_states[:, 0]))

        # Filter label tokens
        label_mask = lmask > 0
        if not label_mask.any():
            empty = torch.empty(0, 1, device=device)
            empty_idx = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty_idx, empty_idx

        # Project label tokens
        label_hidden = self.dropout(self.label_projector(hidden_states[label_mask]))

        # Get batch and label IDs for each token
        batch_indices_all = (
            torch.arange(batch_size, device=device)
            .unsqueeze(1)
            .expand(batch_size, seq_len)
        )
        token_batch_ids = batch_indices_all[label_mask]
        token_label_ids = lmask[label_mask].long()

        # Aggregate by (batch, label_id) using scatter
        max_label_id = int(token_label_ids.max().item())
        num_slots = batch_size * max_label_id

        # Compute flat indices: batch_id * max_label_id + (label_id - 1)
        flat_indices = token_batch_ids * max_label_id + (token_label_ids - 1)

        # Aggregate label representations (mean pooling)
        projected_dim = label_hidden.shape[-1]
        aggregated = torch.zeros(num_slots, projected_dim, device=device)
        counts = torch.zeros(num_slots, device=device)

        aggregated.index_add_(0, flat_indices, label_hidden)
        counts.index_add_(0, flat_indices, torch.ones(len(flat_indices), device=device))

        # Keep only non-empty slots
        valid_mask = counts > 0
        if not valid_mask.any():
            empty = torch.empty(0, 1, device=device)
            empty_idx = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty_idx, empty_idx

        aggregated = aggregated[valid_mask] / counts[valid_mask].unsqueeze(-1)

        # Reconstruct batch and label IDs
        all_batch_ids = (
            torch.arange(batch_size, device=device)
            .unsqueeze(1)
            .expand(batch_size, max_label_id)
            .reshape(-1)[valid_mask]
        )
        all_label_ids = (
            torch.arange(1, max_label_id + 1, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .reshape(-1)[valid_mask]
        )

        # Compute similarities
        cls_expanded = cls_repr[all_batch_ids]
        logits = self.similarity_head(cls_expanded, aggregated)

        return logits, all_batch_ids, all_label_ids


class GliZNetLoss(nn.Module):
    """Computes training loss for GliZNet."""

    def __init__(self, config: GliZNetConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        batch_indices: torch.Tensor,
        label_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss.

        Args:
            logits: Predicted scores (N, 1)
            labels: Ground truth labels (B, MaxLabels)
            batch_indices: Batch index for each logit (N,)
            label_ids: Label ID for each logit (N,)

        Returns:
            Combined loss scalar
        """
        # Gather targets
        target_indices = label_ids - 1
        targets = labels[batch_indices, target_indices].float().view(-1, 1)

        # Filter padding
        valid_mask = targets != -100
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        valid_logits = logits[valid_mask.view(-1)]
        valid_targets = targets[valid_mask].view(-1, 1)
        valid_batch_ids = batch_indices[valid_mask.view(-1)]

        # Binary cross-entropy loss
        bce_loss = (
            F.binary_cross_entropy_with_logits(
                valid_logits, valid_targets, reduction="mean"
            )
            * self.config.scale_loss
        )

        total_loss = bce_loss

        # Contrastive loss
        probs = valid_logits.sigmoid()
        if self.config.contrastive_loss_weight > 0:
            contrastive = self._contrastive_loss(probs, valid_targets, valid_batch_ids)
            total_loss = total_loss + contrastive

        # Separation loss
        if self.config.separation_loss_weight > 0:
            separation = self._separation_loss(probs, valid_targets)
            total_loss = total_loss + separation

        return total_loss

    def _contrastive_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Contrastive loss: push positive/negative logits apart."""
        batch_size = batch_indices.max() + 1
        logits_flat = logits.view(-1)
        labels_flat = labels.view(-1)

        pos_mask = labels_flat > 0.5
        neg_mask = labels_flat <= 0.5

        # Find min positive and max negative per sample
        min_pos = torch.full((batch_size,), float("inf"), device=logits.device)
        max_neg = torch.full((batch_size,), float("-inf"), device=logits.device)

        if pos_mask.any():
            min_pos.scatter_reduce_(
                0,
                batch_indices[pos_mask],
                logits_flat[pos_mask],
                reduce="amin",
                include_self=False,
            )
        if neg_mask.any():
            max_neg.scatter_reduce_(
                0,
                batch_indices[neg_mask],
                logits_flat[neg_mask],
                reduce="amax",
                include_self=False,
            )

        # Compute margin violation
        valid = (min_pos < float("inf")) & (max_neg > float("-inf"))
        if not valid.any():
            return torch.tensor(0.0, device=logits.device)

        violations = F.relu(self.config.margin + max_neg[valid] - min_pos[valid])

        # Temperature scaling
        avg_labels = logits.shape[0] / max(valid.sum(), 1)
        temperature = self.config.temperature * (
            self.config.temperature_scale_base / max(avg_labels, 1)
        )

        return violations.sum() * temperature * self.config.contrastive_loss_weight

    def _separation_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Separation loss: encourage logits to be above/below thresholds."""
        if self.config.separation_loss_weight == 0:
            return torch.tensor(0.0, device=logits.device)
        logits = logits.view(-1)
        labels = labels.view(-1)

        pos_mask = labels > 0.5
        neg_mask = labels <= 0.5

        loss = torch.tensor(0.0, device=logits.device)

        if pos_mask.any():
            pos_logits = logits[pos_mask]
            loss = loss + F.relu(self.config.positive_logit_margin - pos_logits).mean()

        if neg_mask.any():
            neg_logits = logits[neg_mask]
            loss = loss + F.relu(neg_logits - self.config.negative_logit_margin).mean()

        return loss * self.config.separation_loss_weight


# ============================================================================
# Main Model
# ============================================================================


class GliZNetPreTrainedModel(PreTrainedModel):
    """Base class for GliZNet models."""

    config_class = GliZNetConfig
    base_model_prefix = "gliznet"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize weights using parent model's initialization."""
        if hasattr(self, "backbone") and hasattr(self.backbone, "_init_weights"):
            self.backbone._init_weights(module)


class GliZNetForSequenceClassification(GliZNetPreTrainedModel):
    """GliZNet model for zero-shot sequence classification.

    Architecture:
        - Backbone transformer (e.g., DeBERTa)
        - Separate projectors for text ([CLS]) and labels
        - Label aggregation via mean pooling
        - Similarity computation (dot product or bilinear)
    """

    def __init__(self, config: GliZNetConfig):
        super().__init__(config)
        self.config = config

        # Load backbone model from config
        if isinstance(config.backbone_config, dict):
            # Get the config class from model_type and instantiate
            backbone_config = AutoConfig.for_model(**config.backbone_config)
        else:
            backbone_config = config.backbone_config

        self.backbone = AutoModel.from_config(backbone_config)
        hidden_size = backbone_config.hidden_size
        projected_dim = config.projected_dim or hidden_size

        # Build projection layers (stored only in aggregator to avoid shared tensors)
        text_projector = self._build_projector(hidden_size, projected_dim)
        label_projector = self._build_projector(hidden_size, projected_dim)

        # Build task-specific components
        similarity_head = SimilarityHead(config, projected_dim)
        self.aggregator = LabelAggregator(
            config, text_projector, label_projector, similarity_head
        )
        self.loss_fn = GliZNetLoss(config)

        # Initialize weights
        self.post_init()

    def _build_projector(self, input_dim: int, output_dim: int) -> nn.Module:
        """Build a projection layer with optional LayerNorm."""
        if input_dim == output_dim and not self.config.use_projection_layernorm:
            return nn.Identity()

        layers = [nn.Linear(input_dim, output_dim)]
        if self.config.use_projection_layernorm:
            layers.append(nn.LayerNorm(output_dim))

        return nn.Sequential(*layers)

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        """Resize token embeddings (for custom tokens)."""
        embeddings = self.backbone.resize_token_embeddings(new_num_tokens)
        # Update config to reflect new vocab size
        self.config.backbone_config["vocab_size"] = new_num_tokens
        return embeddings

    @classmethod
    def from_backbone_pretrained(
        cls,
        config: GliZNetConfig,
        **kwargs,
    ) -> "GliZNetForSequenceClassification":
        """Create a new GliZNet model with pretrained backbone weights.

        Use this method when creating a NEW model (not loading a saved one).
        The backbone will be initialized with pretrained weights.

        Args:
            config: GliZNetConfig with backbone_model specified
            **kwargs: Additional arguments for backbone loading

        Returns:
            GliZNet model with pretrained backbone
        """
        # Create model (backbone initialized randomly via from_config)
        model = cls(config)

        # Load pretrained backbone weights
        pretrained_backbone = AutoModel.from_pretrained(config.backbone_model, **kwargs)
        model.backbone.load_state_dict(pretrained_backbone.state_dict())

        return model

    @classmethod
    def from_pretrained_with_tokenizer(
        cls,
        pretrained_path: str,
        tokenizer: "GliZNETTokenizer",
        **kwargs,
    ) -> "GliZNetForSequenceClassification":
        """Load pretrained model and resize embeddings for tokenizer.

        Args:
            pretrained_path: Path to saved model
            tokenizer: GliZNETTokenizer instance
            **kwargs: Additional arguments for from_pretrained

        Returns:
            Model with resized embeddings
        """
        model = cls.from_pretrained(pretrained_path, **kwargs)
        model.resize_token_embeddings(len(tokenizer))
        return model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        lmask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_stats: bool = False,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Tuple, GliZNetOutput]:
        """Forward pass.

        Args:
            input_ids: Input token IDs (B, L)
            attention_mask: Attention mask (B, L)
            lmask: Label mask, >0 for label tokens (B, L)
            labels: Ground truth labels (B, MaxLabels), -100 for padding
            return_dict: Whether to return a dict or tuple

        Returns:
            GliZNetOutput or tuple of (loss, logits, batch_indices, label_ids)
        """
        # Encode with backbone
        encoder_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_states = encoder_outputs.last_hidden_state

        # Aggregate labels and compute similarities
        logits, batch_indices, label_ids = self.aggregator(hidden_states, lmask)

        # Compute loss
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels, batch_indices, label_ids)

        if not return_dict:
            return (loss, logits, batch_indices, label_ids)

        return GliZNetOutput(
            loss=loss,
            logits=logits,
            batch_indices=batch_indices if return_stats else None,
            label_ids=label_ids if return_stats else None,
        )

    @torch.inference_mode()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        lmask: torch.Tensor,
        activation: Literal["sigmoid", "softmax"] = "sigmoid",
    ) -> List[List[float]]:
        """Predict label scores for each sample.

        Args:
            input_ids: Input token IDs (B, L)
            attention_mask: Attention mask (B, L)
            lmask: Label mask (B, L)
            activation: Activation function to apply

        Returns:
            List of score lists, one per sample
        """
        self.eval()

        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            lmask=lmask,
            return_stats=True,
            return_dict=True,
        )

        logits = outputs.logits.squeeze(-1)
        batch_indices = outputs.batch_indices
        label_ids = outputs.label_ids

        # Apply activation
        if activation == "sigmoid":
            scores = torch.sigmoid(logits)
        else:
            scores = logits

        # Group by batch
        batch_size = input_ids.shape[0]
        results = [[] for _ in range(batch_size)]

        scores_list = scores.cpu().tolist()
        batch_list = batch_indices.cpu().tolist()
        label_list = label_ids.cpu().tolist()

        for batch_id in range(batch_size):
            # Collect scores for this batch
            batch_scores = [
                (label_id, score)
                for b, label_id, score in zip(batch_list, label_list, scores_list)
                if b == batch_id
            ]

            if batch_scores:
                # Sort by label_id
                batch_scores.sort(key=lambda x: x[0])
                final_scores = [score for _, score in batch_scores]

                # Apply softmax if requested
                if activation == "softmax":
                    final_scores = torch.softmax(
                        torch.tensor(final_scores), dim=0
                    ).tolist()

                results[batch_id] = final_scores

        return results
