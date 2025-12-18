from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModel, BertConfig, BertPreTrainedModel
from transformers.modeling_outputs import ModelOutput

SimilarityMetric = Literal["dot", "bilinear", "dot_learning"]

@dataclass
class GliZNetOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[Union[List[torch.FloatTensor], torch.Tensor]] = None
    hidden_states: Optional[torch.FloatTensor] = None
    batch_indices: Optional[torch.Tensor] = None
    label_ids: Optional[torch.Tensor] = None


class GliZNetSimilarityHead(nn.Module):
    def __init__(self, config, projected_dim):
        super().__init__()
        self.config = config
        if self.config.similarity_metric == "bilinear":
            self.classifier = nn.Bilinear(projected_dim, projected_dim, 1)
        elif self.config.similarity_metric == "dot_learning":
            self.classifier = nn.Linear(projected_dim, 1)
        else:
            self.classifier = None

    def forward(self, text_repr, label_repr):
        if self.config.similarity_metric == "dot":
            return (text_repr * label_repr).mean(dim=1, keepdim=True)
        elif self.config.similarity_metric == "bilinear":
            return self.classifier(text_repr, label_repr)
        return self.classifier(text_repr * label_repr)


class GliZNetRepresentationAggregator(nn.Module):
    def __init__(self, config, cls_proj, label_proj, similarity_head):
        super().__init__()
        self.config = config
        self.cls_proj = cls_proj
        self.label_proj = label_proj
        self.similarity_head = similarity_head

    def forward(self, hidden_states, lmask, cls_attn_weights=None):
        if getattr(self.config, "use_separator_pooling", False):
            return self._compute_with_separator_tokens(hidden_states, lmask)
        return self._compute_original(hidden_states, lmask, cls_attn_weights)

    def _compute_with_separator_tokens(self, hidden_states, lmask):
        cls_tokens = self.cls_proj(hidden_states[:, 0])
        separator_mask = lmask.bool()
        separator_positions = separator_mask.nonzero()
        separator_hidden = self.label_proj(hidden_states[separator_mask])
        separator_batch_indices = separator_positions[:, 0]
        label_ids = lmask[separator_mask]

        cls_for_separators = cls_tokens[separator_batch_indices]
        logits = self.similarity_head(cls_for_separators, separator_hidden)

        return logits, separator_batch_indices, label_ids

    def _compute_original(self, hidden_states, lmask, cls_attn_weights):
        cls_tokens = self.cls_proj(hidden_states[:, 0])
        projected_dim = cls_tokens.shape[-1]
        label_mask = lmask > 0

        label_positions = label_mask.nonzero()
        label_hidden = self.label_proj(hidden_states[label_mask])
        label_ids = lmask[label_mask]
        label_batch_indices = label_positions[:, 0]

        pairs = torch.stack([label_batch_indices, label_ids], dim=1)
        unique_pairs, inverse_indices = torch.unique(pairs, dim=0, return_inverse=True)

        num_unique_labels = unique_pairs.shape[0]

        if cls_attn_weights is not None:
            attn_weights = cls_attn_weights[label_mask]
            attn_sums = torch.zeros(num_unique_labels, device=hidden_states.device)
            attn_sums.scatter_add_(0, inverse_indices, attn_weights)
            attn_sums = attn_sums.clamp_min(1e-8)

            weighted_hidden = label_hidden * attn_weights.unsqueeze(-1)
            label_representations = torch.zeros(
                num_unique_labels, projected_dim, device=hidden_states.device
            )
            label_representations.scatter_add_(
                0,
                inverse_indices.unsqueeze(-1).expand(-1, projected_dim),
                weighted_hidden,
            )
            label_representations = label_representations / attn_sums.unsqueeze(-1)
        else:
            counts = torch.zeros(num_unique_labels, device=hidden_states.device)
            counts.scatter_add_(
                0, inverse_indices, torch.ones_like(inverse_indices, dtype=torch.float)
            )

            label_representations = torch.zeros(
                num_unique_labels, projected_dim, device=hidden_states.device
            )
            label_representations.scatter_add_(
                0,
                inverse_indices.unsqueeze(-1).expand(-1, projected_dim),
                label_hidden,
            )
            label_representations = label_representations / counts.unsqueeze(
                -1
            ).clamp_min(1)

        batch_indices = unique_pairs[:, 0]
        out_label_ids = unique_pairs[:, 1]

        cls_for_labels = cls_tokens[batch_indices]
        logits = self.similarity_head(cls_for_labels, label_representations)

        return logits, batch_indices, out_label_ids


class GliZNetLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.margin = config.margin

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, batch_indices: torch.Tensor, label_ids: torch.Tensor):
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
            F.binary_cross_entropy_with_logits(valid_logits, valid_targets, reduction=None).mean() * self.config.scale_loss
        )

        # Contrastive Loss
        contrastive_loss = torch.tensor(0.0, device=logits.device)
        if self.config.contrastive_loss_weight > 0:
            valid_batch_indices = batch_indices[valid_mask.view(-1)]
            unique_batches, counts = torch.unique(
                valid_batch_indices, return_counts=True
            )

            logits_splits = torch.split(torch.sigmoid(valid_logits), counts.tolist())
            targets_splits = torch.split(valid_targets, counts.tolist())

            # Calculate temperature scaling based on average number of valid labels
            avg_valid_labels = valid_logits.shape[0] / max(len(unique_batches), 1)
            temperature = self.config.temperature * (10.0 / max(avg_valid_labels, 1))

            contrastive_loss = (
                sum(map(self._compute_contrastive_loss, logits_splits, targets_splits))
                * temperature
                * self.config.contrastive_loss_weight
            )

        loss = bce_loss + contrastive_loss

        if self.config.barlow_loss_weight > 0:
            loss = loss + self.config.barlow_loss_weight * self._compute_barlow_loss(
                valid_logits
            )

        return loss

    def _compute_contrastive_loss(self, logits, labels):
        positive_logits = logits[labels == 1.0]
        negative_logits = logits[labels != 1.0]

        if positive_logits.numel() == 0 or negative_logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device)

        min_pos = positive_logits.min()
        max_neg = negative_logits.max()
        return F.relu(self.margin + max_neg - min_pos)

    def _compute_barlow_loss(self, logits, coef=0.005):
        if logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device)
        z = logits.unsqueeze(0) if logits.dim() == 1 else logits
        z_norm = (z - z.mean(dim=0)) / (z.std(dim=0) + 1e-6)
        c = torch.mm(z_norm.T, z_norm) / z_norm.shape[0]
        off_diag = c.flatten()[1:].view(c.size(0), -1).pow_(2).sum()
        return coef * off_diag


def create_gli_znet_for_sequence_classification(base_class=BertPreTrainedModel):
    class GliZNetForSequenceClassification(base_class):
        def __init__(
            self,
            config: BertConfig,
            projected_dim: Optional[int] = None,
            similarity_metric: str = "dot",
            temperature: float = 1.0,
            dropout_rate: float = 0.1,
            scale_loss: float = 10.0,
            margin: float = 0.1,
            barlow_loss_weight: float = 0.1,
            contrastive_loss_weight: float = 1.0,
            use_separator_pooling: bool = False,
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
                barlow_loss_weight: Weight for Barlow Twins regularization loss
                contrastive_loss_weight: Weight for hard negative mining loss
                use_separator_pooling: If True, use separator token embeddings directly;
                    if False, average label token embeddings (requires custom separator token)
            """
            super().__init__(config)
            if similarity_metric not in ["dot", "bilinear", "dot_learning"]:
                raise ValueError(
                    f"Unsupported similarity metric: {similarity_metric}. Supported: 'dot', 'bilinear', 'dot_learning'."
                )
            setattr(self, self.base_model_prefix, AutoModel.from_config(config=config))
            self._initialize_config(
                config,
                projected_dim,
                similarity_metric,
                temperature,
                dropout_rate,
                barlow_loss_weight,
                contrastive_loss_weight,
                use_separator_pooling,
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
            tokenizer,  # GliZNETTokenizer instance
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
            config,
            projected_dim,
            similarity_metric,
            temperature,
            dropout_rate,
            barlow_loss_weight,
            contrastive_loss_weight,
            use_separator_pooling,
        ):
            config.projected_dim = getattr(config, "projected_dim", projected_dim)
            config.similarity_metric = getattr(
                config, "similarity_metric", similarity_metric
            )
            config.temperature = getattr(config, "temperature", temperature)
            config.dropout_rate = getattr(config, "dropout_rate", dropout_rate)
            config.barlow_loss_weight = getattr(
                config, "barlow_loss_weight", barlow_loss_weight
            )
            config.contrastive_loss_weight = getattr(
                config, "contrastive_loss_weight", contrastive_loss_weight
            )
            config.use_separator_pooling = getattr(
                config, "use_separator_pooling", use_separator_pooling
            )
            self.config = config

        def _setup_layers(self):
            projected_dim = self.config.projected_dim or self.config.hidden_size

            # Separate projectors for CLS and label tokens
            self.cls_proj = (
                nn.Linear(self.config.hidden_size, projected_dim)
                if projected_dim != self.config.hidden_size
                else nn.Identity()
            )
            self.label_proj = (
                nn.Linear(self.config.hidden_size, projected_dim)
                if projected_dim != self.config.hidden_size
                else nn.Identity()
            )

            self.similarity_head = GliZNetSimilarityHead(self.config, projected_dim)
            self.aggregator = GliZNetRepresentationAggregator(
                self.config, self.cls_proj, self.label_proj, self.similarity_head
            )
            self.loss_module = GliZNetLoss(self.config)

        def backbone_forward(self, *args, **kwargs):
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
                batch_indices=batch_indices,
                label_ids=label_ids,
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
            return self.dropout(encoder_outputs.last_hidden_state), cls_attn_weights

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
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                lmask=lmask,
                return_dict=True,
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
            batch_scores = {}
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
