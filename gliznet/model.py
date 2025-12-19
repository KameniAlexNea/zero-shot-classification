from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModel, DebertaV2Config, DebertaV2PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, ModelOutput

if TYPE_CHECKING:
    from .tokenizer import GliZNETTokenizer

SimilarityMetric = Literal["dot", "bilinear"]


class GlizNetConfig(DebertaV2Config):
    margin: float = 0.5
    projected_dim: Optional[int] = None
    similarity_metric: SimilarityMetric = "dot"
    temperature: float = 1.0
    dropout_rate: float = 0.1
    scale_loss: float = 10.0
    contrastive_loss_weight: float = 1.0
    use_separator_pooling: bool = False
    temperature_scale_base: float = 10.0
    use_projection_layernorm: bool = True
    separation_loss_weight: float = 0.5
    positive_logit_margin: float = 1.0
    negative_logit_margin: float = -1.0


@dataclass
class GliZNetOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[Union[List[torch.FloatTensor], torch.Tensor]] = None
    hidden_states: Optional[torch.FloatTensor] = None
    batch_indices: Optional[torch.Tensor] = None
    label_ids: Optional[torch.Tensor] = None


class GliZNetSimilarityHead(nn.Module):
    def __init__(self, config: GlizNetConfig, projected_dim):
        super().__init__()
        self.config = config
        if self.config.similarity_metric == "bilinear":
            self.classifier = nn.Bilinear(projected_dim, projected_dim, 1)
        else:  # dot
            self.classifier = nn.Linear(projected_dim, 1)

    def forward(self, text_repr, label_repr):
        if self.config.similarity_metric == "bilinear":
            return self.classifier(text_repr, label_repr)
        return self.classifier(text_repr * label_repr)


class GliZNetRepresentationAggregator(nn.Module):
    def __init__(self, config: GlizNetConfig, cls_proj, label_proj, similarity_head):
        super().__init__()
        self.config = config
        self.cls_proj = cls_proj
        self.label_proj = label_proj
        self.similarity_head = similarity_head
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        lmask: torch.Tensor,
        cls_attn_weights: Optional[torch.Tensor] = None,
    ):
        cls_tokens = self.dropout(self.cls_proj(hidden_states[:, 0]))
        label_mask = lmask > 0

        if not label_mask.any():
            empty_logits = torch.empty(0, 1, device=hidden_states.device)
            empty_indices = torch.empty(
                0, dtype=torch.long, device=hidden_states.device
            )
            return empty_logits, empty_indices, empty_indices

        label_hidden = self.dropout(self.label_proj(hidden_states[label_mask]))
        use_attn_weights = cls_attn_weights is not None and not getattr(
            self.config, "use_separator_pooling", False
        )
        label_weights = cls_attn_weights[label_mask] if use_attn_weights else None

        return self._scatter_aggregate(
            cls_tokens,
            lmask,
            label_mask,
            label_hidden,
            label_weights,
        )

    def _scatter_aggregate(
        self,
        cls_tokens: torch.Tensor,
        lmask: torch.Tensor,
        label_mask: torch.Tensor,
        label_hidden: torch.Tensor,
        label_weights: Optional[torch.Tensor],
    ):
        batch_size, seq_len = lmask.shape
        device = lmask.device
        max_label_id = int(lmask.max().item())

        if max_label_id == 0:
            empty_logits = torch.empty(0, 1, device=device)
            empty_indices = torch.empty(0, dtype=torch.long, device=device)
            return empty_logits, empty_indices, empty_indices

        batch_indices = (
            torch.arange(batch_size, device=device)
            .unsqueeze(1)
            .expand(batch_size, seq_len)
        )
        token_batch_indices = batch_indices[label_mask]
        label_ids = lmask[label_mask].long()

        token_weights = (
            label_hidden.new_ones(label_hidden.shape[0])
            if label_weights is None
            else label_weights.to(label_hidden.dtype)
        )

        projected_dim = label_hidden.shape[-1]
        total_slots = batch_size * max_label_id
        scatter_indices = token_batch_indices * max_label_id + (label_ids - 1)

        aggregated = torch.zeros(total_slots, projected_dim, device=device)
        weight_sums = torch.zeros(total_slots, device=device)

        aggregated.index_add_(
            0, scatter_indices, label_hidden * token_weights.unsqueeze(-1)
        )
        weight_sums.index_add_(0, scatter_indices, token_weights)

        present_mask = weight_sums > 0
        if not present_mask.any():
            empty_logits = torch.empty(0, 1, device=device)
            empty_indices = torch.empty(0, dtype=torch.long, device=device)
            return empty_logits, empty_indices, empty_indices

        aggregated = aggregated / weight_sums.clamp_min(1e-8).unsqueeze(-1)

        all_batch_ids = (
            torch.arange(batch_size, device=device)
            .unsqueeze(1)
            .expand(batch_size, max_label_id)
            .reshape(-1)
        )
        all_label_ids = (
            torch.arange(1, max_label_id + 1, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .reshape(-1)
        )

        final_batch_indices = all_batch_ids[present_mask]
        final_label_ids = all_label_ids[present_mask]
        final_representations = aggregated[present_mask]

        cls_for_labels = cls_tokens[final_batch_indices]
        logits = self.similarity_head(cls_for_labels, final_representations)

        return logits, final_batch_indices, final_label_ids


class GliZNetLoss(nn.Module):
    def __init__(self, config: GlizNetConfig):
        super().__init__()
        self.config = config
        self.margin = config.margin

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        batch_indices: torch.Tensor,
        label_ids: torch.Tensor,
    ):
        if labels is None:
            return None

        # label_ids are 1-based, convert to 0-based index
        target_indices = label_ids - 1

        # Gather targets
        # labels: (B, MaxLabels)
        targets = labels[batch_indices, target_indices].float().view(-1, 1)

        # Filter out padding (-100)
        valid_mask = targets != -100
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        valid_logits = logits[valid_mask.view(-1)]
        valid_targets = targets[valid_mask].view(-1, 1)

        # BCE Loss
        bce_loss: torch.Tensor = (
            F.binary_cross_entropy_with_logits(
                valid_logits, valid_targets, reduction="none"
            ).mean()
            * self.config.scale_loss
        )

        # Contrastive Loss
        contrastive_loss = torch.tensor(0.0, device=logits.device)
        if self.config.contrastive_loss_weight > 0:
            valid_batch_indices = batch_indices[valid_mask.view(-1)]
            contrastive_loss = self._compute_contrastive_loss(
                valid_logits, valid_targets, valid_batch_indices
            )

        loss = bce_loss + contrastive_loss

        if self.config.separation_loss_weight > 0:
            separation_loss = self._compute_separation_loss(valid_logits, valid_targets)
            loss = loss + self.config.separation_loss_weight * separation_loss

        return loss

    def _compute_contrastive_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, batch_indices: torch.Tensor
    ) -> torch.Tensor:
        """Compute vectorized contrastive loss across batches.

        Args:
            logits: Valid logits (N, 1)
            labels: Valid labels (N, 1)
            batch_indices: Batch index for each logit (N,)

        Returns:
            Scalar contrastive loss tensor
        """
        batch_size = batch_indices.max() + 1

        # Vectorized computation without splits
        logits_flat = logits.view(-1)
        labels_flat = labels.view(-1)

        pos_mask = labels_flat > 0.5
        neg_mask = labels_flat <= 0.5

        # Initialize with extreme values
        min_pos_per_sample = torch.full(
            (batch_size,), float("inf"), device=logits.device
        )
        max_neg_per_sample = torch.full(
            (batch_size,), float("-inf"), device=logits.device
        )

        # Scatter min for positives and max for negatives
        if pos_mask.any():
            min_pos_per_sample.scatter_reduce_(
                0,
                batch_indices[pos_mask],
                logits_flat[pos_mask],
                reduce="amin",
                include_self=False,
            )
        if neg_mask.any():
            max_neg_per_sample.scatter_reduce_(
                0,
                batch_indices[neg_mask],
                logits_flat[neg_mask],
                reduce="amax",
                include_self=False,
            )

        # Only compute loss for samples that have both positive and negative labels
        valid_samples = (min_pos_per_sample < float("inf")) & (
            max_neg_per_sample > float("-inf")
        )

        if not valid_samples.any():
            return torch.tensor(0.0, device=logits.device)

        sample_losses = F.relu(
            self.margin
            + max_neg_per_sample[valid_samples]
            - min_pos_per_sample[valid_samples]
        )

        # Temperature scaling based on average labels per sample
        num_valid_samples = valid_samples.sum()
        avg_valid_labels = logits.shape[0] / max(num_valid_samples, 1)
        temperature_scale_base = getattr(self.config, "temperature_scale_base", 10.0)
        temperature = self.config.temperature * (
            temperature_scale_base / max(avg_valid_labels, 1)
        )

        return sample_losses.sum() * temperature * self.config.contrastive_loss_weight

    def _compute_separation_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        if logits.numel() == 0:
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

        return loss


def create_gli_znet_for_sequence_classification(base_class=DebertaV2PreTrainedModel):
    class GliZNetForSequenceClassification(base_class):
        def __init__(
            self,
            config: GlizNetConfig,
            projected_dim: Optional[int] = None,
            similarity_metric: str = "dot",
            temperature: float = 1.0,
            dropout_rate: float = 0.1,
            scale_loss: float = 10.0,
            margin: float = 0.1,
            contrastive_loss_weight: float = 1.0,
            use_separator_pooling: bool = True,
            temperature_scale_base: float = 10.0,
            use_projection_layernorm: bool = True,
            separation_loss_weight: float = 0.1,
            positive_logit_margin: float = 1.0,
            negative_logit_margin: float = -1.0,
        ):
            """Initialize GliZNet model.

            Args:
                config: BERT configuration
                projected_dim: Dimension for projection layers (None = use hidden_size)
                similarity_metric: How to compute similarity ('dot', 'bilinear', 'dot_learning')
                temperature: Temperature for contrastive loss scaling
                dropout_rate: Dropout probability
                scale_loss: Multiplier for BCE loss
                margin: Margin for contrastive loss
                contrastive_loss_weight: Weight for hard negative mining loss
                use_separator_pooling: If True, use separator token embeddings directly;
                    if False, average label token embeddings (requires custom separator token)
                temperature_scale_base: Base value for temperature scaling
                use_projection_layernorm: Whether to apply LayerNorm after projection
                separation_loss_weight: Weight for logit separation regularization
                positive_logit_margin: Minimum desired logit for positive labels
                negative_logit_margin: Maximum desired logit for negative labels
            """
            super().__init__(config)
            if similarity_metric not in ["dot", "bilinear"]:
                raise ValueError(
                    f"Unsupported similarity metric: {similarity_metric}. Supported: 'dot', 'bilinear'."
                )
            setattr(self, self.base_model_prefix, AutoModel.from_config(config=config))
            self._initialize_config(
                config,
                projected_dim,
                similarity_metric,
                temperature,
                dropout_rate,
                contrastive_loss_weight,
                use_separator_pooling,
                temperature_scale_base,
                use_projection_layernorm,
                margin=margin,
                separation_loss_weight=separation_loss_weight,
                positive_logit_margin=positive_logit_margin,
                negative_logit_margin=negative_logit_margin,
                scale_loss=scale_loss,
            )

            self.scale_loss = scale_loss
            self.margin = margin
            self.dropout = nn.Dropout(self.config.dropout_rate)
            self._setup_layers()
            self.post_init()

        def resize_token_embeddings(
            self, new_num_tokens: Optional[int] = None
        ) -> nn.Embedding:
            """
            Resize input token embeddings matrix of the model if new_num_tokens != config.vocab_size.

            This method should be called when custom tokens are added to the tokenizer.
            """
            base_model = getattr(self, self.base_model_prefix)
            return base_model.resize_token_embeddings(new_num_tokens)

        @classmethod
        def from_pretrained_with_tokenizer(
            cls,
            pretrained_model_name_or_path: str,
            tokenizer: "GliZNETTokenizer",  # GliZNETTokenizer instance
            **kwargs,
        ):
            """
            Create model from pretrained and automatically resize embeddings if tokenizer has custom tokens.

            Args:
                pretrained_model_name_or_path: Path to pretrained model
                tokenizer: GliZNETTokenizer instance
                **kwargs: Additional arguments for model initialization
            """
            model = cls.from_pretrained(pretrained_model_name_or_path, **kwargs)

            # Resize token embeddings if custom tokens were added
            if tokenizer.has_custom_tokens():
                new_vocab_size = tokenizer.get_vocab_size()
                model.resize_token_embeddings(new_vocab_size)
                model.config.vocab_size = new_vocab_size
                # Use separator pooling when custom separator tokens are present
                model.config.use_separator_pooling = True

            return model

        def _initialize_config(
            self,
            config: GlizNetConfig,
            projected_dim,
            similarity_metric,
            temperature,
            dropout_rate,
            contrastive_loss_weight,
            use_separator_pooling,
            temperature_scale_base,
            use_projection_layernorm,
            margin,
            separation_loss_weight,
            positive_logit_margin,
            negative_logit_margin,
            scale_loss,
        ):
            config.projected_dim = getattr(config, "projected_dim", projected_dim)
            config.similarity_metric = getattr(
                config, "similarity_metric", similarity_metric
            )
            config.temperature = getattr(config, "temperature", temperature)
            config.dropout_rate = getattr(config, "dropout_rate", dropout_rate)
            config.contrastive_loss_weight = getattr(
                config, "contrastive_loss_weight", contrastive_loss_weight
            )
            config.use_separator_pooling = getattr(
                config, "use_separator_pooling", use_separator_pooling
            )
            config.temperature_scale_base = getattr(
                config, "temperature_scale_base", temperature_scale_base
            )
            config.use_projection_layernorm = getattr(
                config, "use_projection_layernorm", use_projection_layernorm
            )
            config.margin = getattr(config, "margin", margin)
            config.separation_loss_weight = getattr(
                config, "separation_loss_weight", separation_loss_weight
            )
            config.positive_logit_margin = getattr(
                config, "positive_logit_margin", positive_logit_margin
            )
            config.negative_logit_margin = getattr(
                config, "negative_logit_margin", negative_logit_margin
            )
            config.scale_loss = getattr(config, "scale_loss", scale_loss)
            self.config = config

        def _setup_layers(self):
            projected_dim = self.config.projected_dim or self.config.hidden_size
            use_ln = getattr(self.config, "use_projection_layernorm", True)

            # Separate projectors for CLS and label tokens
            if projected_dim != self.config.hidden_size or use_ln:
                cls_proj = nn.Sequential(
                    nn.Linear(self.config.hidden_size, projected_dim),
                    nn.LayerNorm(projected_dim) if use_ln else nn.Identity(),
                )
                label_proj = nn.Sequential(
                    nn.Linear(self.config.hidden_size, projected_dim),
                    nn.LayerNorm(projected_dim) if use_ln else nn.Identity(),
                )
            else:
                cls_proj = nn.Identity()
                label_proj = nn.Identity()

            similarity_head = GliZNetSimilarityHead(self.config, projected_dim)
            self.aggregator = GliZNetRepresentationAggregator(
                self.config, cls_proj, label_proj, similarity_head
            )
            self.loss_module = GliZNetLoss(self.config)

        def backbone_forward(self, *args, **kwargs) -> BaseModelOutput:
            return getattr(self, self.base_model_prefix)(*args, **kwargs)

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            lmask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            return_indices: bool = False,
        ) -> Union[Tuple[torch.Tensor], GliZNetOutput]:
            return_dict = (
                return_dict if return_dict is not None else self.config.use_return_dict
            )

            if lmask is None:
                raise ValueError(
                    "lmask is required for GliZNetForSequenceClassification"
                )

            hidden_states, attentions = self._get_hidden_states(
                input_ids,
                attention_mask,
                token_type_ids,
                output_attentions,
                output_hidden_states,
            )

            logits, batch_indices, label_ids = self.aggregator(
                hidden_states, lmask, attentions
            )

            loss = None
            if labels is not None:
                loss = self.loss_module(logits, labels, batch_indices, label_ids)

            if not return_dict:
                output = (logits, None)
                return ((loss,) + output) if loss is not None else output

            return GliZNetOutput(
                loss=loss,
                logits=logits,
                hidden_states=None,
                batch_indices=batch_indices if return_indices else None,
                label_ids=label_ids if return_indices else None,
            )

        def _get_hidden_states(
            self,
            input_ids,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            # Only compute attention weights if explicitly requested
            compute_attention = output_attentions is not None and output_attentions
            encoder_outputs = self.backbone_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=compute_attention,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            cls_attn_weights = None
            if compute_attention and encoder_outputs.attentions is not None:
                cls_attn_weights = encoder_outputs.attentions[-1].mean(dim=1)[:, 0, :]
            return encoder_outputs.last_hidden_state, cls_attn_weights

        @torch.inference_mode()
        def encode(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor
        ) -> torch.Tensor:
            """
            Encode input_ids using the backbone model.
            Returns the last hidden state of the [CLS] token.
            """
            outputs = self.backbone_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            return outputs.last_hidden_state[:, 0]

        @torch.inference_mode()
        def predict(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            lmask: torch.Tensor,
            activation_fn: Optional[str] = "sigmoid",
        ) -> List[List[float]]:
            """Predict label scores for input samples.

            Args:
                input_ids: Token IDs (batch_size, seq_len)
                attention_mask: Attention mask (batch_size, seq_len)
                lmask: Label mask indicating label positions (batch_size, seq_len)
                activation_fn: Activation function to apply ('sigmoid' or 'softmax')

            Returns:
                List of score lists, one per sample in the batch
            """
            self.eval()

            # Get outputs with indices
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                lmask=lmask,
                return_dict=True,
                return_indices=True,
            )

            logits = outputs.logits
            batch_indices = outputs.batch_indices
            label_ids = outputs.label_ids

            # Reconstruct results
            batch_size = input_ids.shape[0]
            results = [[] for _ in range(batch_size)]

            # Apply activation
            if activation_fn == "sigmoid":
                scores = torch.sigmoid(logits.squeeze(-1))
            else:
                scores = logits.squeeze(-1)

            scores = scores.cpu().tolist()
            batch_indices = batch_indices.cpu().tolist()
            label_ids = label_ids.cpu().tolist()

            # Group by batch
            batch_scores: dict[int, List[Tuple[int, float]]] = {}
            for b_idx, l_id, score in zip(batch_indices, label_ids, scores):
                if b_idx not in batch_scores:
                    batch_scores[b_idx] = []
                batch_scores[b_idx].append((l_id, score))

            for b_idx in range(batch_size):
                if b_idx in batch_scores:
                    # Sort by label_id
                    sample_scores = sorted(batch_scores[b_idx], key=lambda x: x[0])
                    final_scores = [s for _, s in sample_scores]

                    if activation_fn == "softmax":
                        final_scores = torch.softmax(
                            torch.tensor(final_scores), dim=0
                        ).tolist()

                    results[b_idx] = final_scores

            return results

    return GliZNetForSequenceClassification


GliZNetForSequenceClassification = create_gli_znet_for_sequence_classification()
