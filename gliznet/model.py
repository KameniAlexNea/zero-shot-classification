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
        backbone_config: Optional[PretrainedConfig] = None,
        projected_dim: Optional[int] = None,
        similarity_metric: Literal["dot", "bilinear", "cosine"] = "cosine",
        dropout_rate: float = 0.1,
        use_projection_layernorm: bool = True,
        # Loss weights
        bce_loss_weight: float = 1.0,
        supcon_loss_weight: float = 1.0,
        label_repulsion_weight: float = 0.1,
        # Temperature/scaling
        logit_scale_init: float = 2.0,  # exp(2) ≈ 7.4 for cosine similarity scaling
        learn_temperature: bool = True,
        # Repulsion settings
        repulsion_threshold: float = 0.3,  # Penalize if cosine sim > this
        # Legacy (kept for backward compatibility)
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

        # New loss configuration
        self.bce_loss_weight = bce_loss_weight
        self.supcon_loss_weight = supcon_loss_weight
        self.label_repulsion_weight = label_repulsion_weight
        self.logit_scale_init = logit_scale_init
        self.learn_temperature = learn_temperature
        self.repulsion_threshold = repulsion_threshold

        # Legacy loss configuration (kept for backward compat)
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
            backbone_config = AutoConfig.from_pretrained(backbone_model)
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
        label_embeddings: Projected label embeddings (for repulsion loss)
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.Tensor] = None
    batch_indices: Optional[torch.Tensor] = None
    label_ids: Optional[torch.Tensor] = None
    label_embeddings: Optional[torch.Tensor] = None


# ============================================================================
# Model Components
# ============================================================================


class SimilarityHead(nn.Module):
    """Computes similarity between text and label representations."""

    def __init__(self, config: GliZNetConfig, projected_dim: int):
        super().__init__()
        self.config = config

        # Learnable temperature for scaling logits (used in SupCon)
        if config.learn_temperature:
            self.logit_scale = nn.Parameter(
                torch.tensor(config.logit_scale_init, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "logit_scale", torch.tensor(config.logit_scale_init, dtype=torch.float32)
            )

        if config.similarity_metric == "bilinear":
            self.classifier = nn.Bilinear(projected_dim, projected_dim, 1)
        elif config.similarity_metric == "dot":
            self.classifier = nn.Linear(projected_dim, 1)
        elif config.similarity_metric == "cosine":
            # For cosine, we just do normalized dot product
            pass
        else:
            raise ValueError(
                f"Unknown similarity_metric: {config.similarity_metric}. "
                "Choose 'dot', 'bilinear', or 'cosine'."
            )

    def forward(
        self, text_repr: torch.Tensor, label_repr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute similarity scores.

        Args:
            text_repr: Text representations (N, D)
            label_repr: Label representations (N, D)

        Returns:
            Tuple of (scaled similarity scores (N, 1), logit_scale)
        """
        if self.config.similarity_metric == "bilinear":
            logits = self.classifier(text_repr, label_repr)
        elif self.config.similarity_metric == "dot":
            logits = self.classifier(text_repr * label_repr)
        else:  # cosine
            # Normalize for cosine similarity
            text_norm = F.normalize(text_repr, p=2, dim=-1)
            label_norm = F.normalize(label_repr, p=2, dim=-1)
            # Cosine similarity: dot product of normalized vectors
            raw_sim = (text_norm * label_norm).sum(dim=-1, keepdim=True)
            # Scale by learnable temperature
            logits = raw_sim * self.logit_scale.exp()

        return logits, self.logit_scale


class LabelAggregator(nn.Module):
    """Aggregates label token embeddings and computes similarities."""

    def __init__(
        self,
        config: GliZNetConfig,
        text_projector: nn.Module,
        label_projector: nn.Module,
        similarity_head: "SimilarityHead",
    ):
        super().__init__()
        self.config = config
        self.text_projector = text_projector
        self.label_projector = label_projector
        self.similarity_head = similarity_head
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self, hidden_states: torch.Tensor, lmask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aggregate label representations and compute similarities.

        Args:
            hidden_states: Encoder outputs (B, L, H)
            lmask: Label mask where >0 indicates label tokens (B, L)

        Returns:
            logits: Similarity scores (N, 1)
            batch_indices: Batch index for each score (N,)
            label_ids: Label ID for each score (N,)
            label_embeddings: Aggregated label embeddings (N, D)
            logit_scale: Current temperature scale
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
            empty_emb = torch.empty(0, cls_repr.shape[-1], device=device)
            dummy_scale = self.similarity_head.logit_scale
            return empty, empty_idx, empty_idx, empty_emb, dummy_scale

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
            empty_emb = torch.empty(0, projected_dim, device=device)
            dummy_scale = self.similarity_head.logit_scale
            return empty, empty_idx, empty_idx, empty_emb, dummy_scale

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
        logits, logit_scale = self.similarity_head(cls_expanded, aggregated)

        return logits, all_batch_ids, all_label_ids, aggregated, logit_scale


class GliZNetLoss(nn.Module):
    """Improved loss for GliZNet with SupCon, label repulsion, and decoupled BCE."""

    def __init__(self, config: GliZNetConfig):
        super().__init__()
        self.config = config
        # Learnable scale specifically for the auxiliary BCE loss (decoupled from SupCon)
        self.bce_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        batch_indices: torch.Tensor,
        label_ids: torch.Tensor,
        label_embeddings: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss with SupCon, repulsion, and BCE.

        Args:
            logits: Predicted scores (N, 1) - already scaled by SimilarityHead
            labels: Ground truth labels (B, MaxLabels)
            batch_indices: Batch index for each logit (N,)
            label_ids: Label ID for each logit (N,)
            label_embeddings: Projected label embeddings (N, D)
            logit_scale: Current temperature scale from SimilarityHead

        Returns:
            Combined loss scalar
        """
        if logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        batch_size = labels.size(0)
        max_label_id = int(label_ids.max().item()) if label_ids.numel() > 0 else 0

        if max_label_id == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Reconstruct dense logits matrix (B, max_labels)
        dense_logits = torch.full(
            (batch_size, max_label_id), float("-inf"), device=logits.device
        )
        col_indices = label_ids - 1
        dense_logits[batch_indices, col_indices] = logits.squeeze(-1)

        # Align targets with dense_logits shape
        valid_cols = min(labels.shape[1], max_label_id)
        current_labels = labels[:, :valid_cols].float()
        if current_labels.shape[1] < max_label_id:
            padding = torch.full(
                (batch_size, max_label_id - current_labels.shape[1]),
                -100.0,
                device=logits.device,
            )
            current_labels = torch.cat([current_labels, padding], dim=1)

        total_loss = torch.tensor(0.0, device=logits.device)

        # --- 1. Supervised Contrastive Loss (Primary) ---
        if self.config.supcon_loss_weight > 0:
            supcon_loss = self._supcon_loss(dense_logits, current_labels)
            total_loss = total_loss + supcon_loss * self.config.supcon_loss_weight

        # --- 2. Label Repulsion Loss (Refined: same-sample only) ---
        if self.config.label_repulsion_weight > 0:
            repulsion_loss = self._label_repulsion_loss(
                label_embeddings, label_ids, batch_indices
            )
            total_loss = total_loss + repulsion_loss * self.config.label_repulsion_weight

        # --- 3. Auxiliary BCE (Decoupled Temperature) ---
        if self.config.bce_loss_weight > 0:
            bce_loss = self._bce_loss(
                dense_logits, current_labels, logit_scale
            )
            total_loss = total_loss + bce_loss * self.config.bce_loss_weight

        return total_loss

    def _supcon_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Supervised Contrastive Loss over all label pairs.

        For each sample, treats positive labels as "anchors" and computes
        softmax over all labels, encouraging high prob for positives.
        """
        mask_valid = targets != -100
        targets_clean = targets.clone()
        targets_clean[~mask_valid] = 0.0

        # Only compute for samples that have at least one positive
        has_positives = (targets_clean > 0.5).any(dim=1)
        if not has_positives.any():
            return torch.tensor(0.0, device=logits.device)

        logits = logits[has_positives]
        targets_clean = targets_clean[has_positives]
        mask_valid_filtered = mask_valid[has_positives]

        # Mask out invalid positions for softmax
        logits_masked = logits.clone()
        logits_masked[~mask_valid_filtered] = float("-inf")

        # Log-softmax over valid labels for each sample
        log_probs = F.log_softmax(logits_masked, dim=1)

        # Compute loss: -mean(log_prob of positive labels)
        pos_mask = (targets_clean > 0.5).float()
        sum_log_prob_pos = (log_probs * pos_mask).sum(dim=1)
        num_pos = pos_mask.sum(dim=1)

        loss_per_sample = -sum_log_prob_pos / (num_pos + 1e-9)
        return loss_per_sample.mean()

    def _label_repulsion_loss(
        self,
        embeddings: torch.Tensor,
        label_ids: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize high cosine similarity between DIFFERENT labels in SAME sample.

        This prevents label embedding collapse while respecting contextual embeddings.
        """
        if embeddings.numel() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        # Normalize embeddings for cosine similarity
        embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
        sim_matrix = torch.matmul(embeddings_norm, embeddings_norm.T)

        # Mask 1: Different labels (label_id_i != label_id_j)
        diff_label_mask = label_ids.unsqueeze(0) != label_ids.unsqueeze(1)

        # Mask 2: Same sample (batch_index_i == batch_index_j)
        # Key insight: only enforce geometry within the same sample's context
        same_batch_mask = batch_indices.unsqueeze(0) == batch_indices.unsqueeze(1)

        # Combine: repel different labels belonging to the same sample
        final_mask = diff_label_mask & same_batch_mask

        if not final_mask.any():
            return torch.tensor(0.0, device=embeddings.device)

        valid_sims = sim_matrix[final_mask]

        # Penalize only if similarity exceeds threshold
        penalties = F.relu(valid_sims - self.config.repulsion_threshold)

        return penalties.mean()

    def _bce_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        main_logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Binary cross-entropy with decoupled temperature.

        Unscales the main logits and applies BCE-specific scaling.
        """
        mask = targets != -100
        if not mask.any():
            return torch.tensor(0.0, device=logits.device)

        valid_logits = logits[mask]
        valid_targets = targets[mask]

        # Decoupling: unscale main logits, then apply BCE-specific scale
        # This prevents SupCon temperature from dominating BCE gradients
        raw_logits = valid_logits / (main_logit_scale.exp() + 1e-9)
        bce_logits = raw_logits * self.bce_scale.abs().clamp(min=0.1)

        return F.binary_cross_entropy_with_logits(
            bce_logits, valid_targets, reduction="mean"
        )


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

        self.backbone: PreTrainedModel = AutoModel.from_config(config.backbone_config)
        hidden_size = config.backbone_config.hidden_size
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
        return self.backbone.resize_token_embeddings(new_num_tokens)

    @classmethod
    def from_backbone_pretrained(
        cls,
        config: GliZNetConfig,
        tokenizer: "GliZNETTokenizer",
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
        pretrained_backbone: PreTrainedModel = AutoModel.from_pretrained(
            config.backbone_model, **kwargs
        )
        model.backbone.load_state_dict(pretrained_backbone.state_dict())
        model.config.backbone_config = pretrained_backbone.config
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
        logits, batch_indices, label_ids, label_embeddings, logit_scale = self.aggregator(
            hidden_states, lmask
        )

        # Compute loss
        loss = None
        if labels is not None:
            loss = self.loss_fn(
                logits=logits,
                labels=labels,
                batch_indices=batch_indices,
                label_ids=label_ids,
                label_embeddings=label_embeddings,
                logit_scale=logit_scale,
            )

        if not return_dict:
            return (loss, logits, batch_indices, label_ids)

        return GliZNetOutput(
            loss=loss,
            logits=logits,
            batch_indices=batch_indices if return_stats else None,
            label_ids=label_ids if return_stats else None,
            label_embeddings=label_embeddings if return_stats else None,
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
