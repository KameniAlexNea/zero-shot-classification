from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModel, BertConfig, BertPreTrainedModel
from transformers.modeling_outputs import ModelOutput


class GliZNetConfig(BertConfig):
    def __init__(
        self,
        projected_dim: Optional[int] = None,
        similarity_metric: str = "dot",
        temperature: float = 1.0,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.projected_dim = projected_dim
        self.similarity_metric = similarity_metric
        self.temperature = temperature
        self.dropout_rate = dropout_rate


@dataclass
class GliZNetOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[Union[List[torch.FloatTensor], torch.Tensor]] = None
    hidden_states: Optional[torch.FloatTensor] = None


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
        ):
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
            )

            self.scale_loss = scale_loss
            self.dropout = nn.Dropout(self.config.dropout_rate)
            self._setup_layers()
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")
            self.post_init()

        def _initialize_config(
            self,
            config,
            projected_dim,
            similarity_metric,
            temperature,
            dropout_rate,
        ):
            config.projected_dim = getattr(config, "projected_dim", projected_dim)
            config.similarity_metric = getattr(
                config, "similarity_metric", similarity_metric
            )
            config.temperature = getattr(config, "temperature", temperature)
            config.dropout_rate = getattr(config, "dropout_rate", dropout_rate)
            self.config = config

        def _setup_layers(self):
            projected_dim = self.config.projected_dim or self.config.hidden_size
            self.proj = (
                nn.Linear(self.config.hidden_size, projected_dim)
                if projected_dim != self.config.hidden_size
                else nn.Identity()
            )

            if self.config.similarity_metric == "bilinear":
                self.classifier = nn.Bilinear(projected_dim, projected_dim, 1)
            elif self.config.similarity_metric == "dot_learning":
                self.classifier = nn.Linear(projected_dim, 1)

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
            outputs_logits = self._compute_batch_logits(
                hidden_states, lmask, attentions
            )
            loss = (
                self._compute_loss(outputs_logits, labels)
                if labels is not None
                else None
            )

            if not return_dict:
                output = (outputs_logits, None)
                return ((loss,) + output) if loss is not None else output

            return GliZNetOutput(loss=loss, logits=outputs_logits, hidden_states=None)

        def compute_similarity(
            self, text_repr: torch.Tensor, label_repr: torch.Tensor
        ) -> torch.Tensor:
            if self.config.similarity_metric == "dot":
                return (text_repr * label_repr).mean(dim=1, keepdim=True)
            elif self.config.similarity_metric == "bilinear":
                return self.classifier(text_repr, label_repr)
            return self.classifier(text_repr * label_repr)

        def _get_hidden_states(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            output_attentions,
            output_hidden_states,
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

        def _compute_batch_logits(
            self,
            hidden_states: torch.Tensor,  # (B, L, H)
            lmask: torch.Tensor,  # (B, L), values: 0 = text, 1... = label groups
            cls_attn_weights: Optional[torch.Tensor],  # (B, L)
        ) -> torch.Tensor:
            """
            Compute logits between [CLS] and averaged label embeddings per sample.
            Fully vectorized implementation - computes similarity only once!
            """
            hidden_proj: torch.Tensor = self.proj(hidden_states)  # (B, L, D)
            projected_dim = hidden_proj.shape[-1]

            # Get [CLS] tokens (first token in each sequence)
            cls_tokens = hidden_proj[:, 0]  # (B, D)

            # Create mask for non-text tokens (label tokens have lmask > 0)
            label_mask = lmask > 0  # (B, L)

            # Get all label tokens and their metadata
            label_positions = (
                label_mask.nonzero()
            )  # (num_label_tokens, 2) - [batch_idx, seq_idx]
            label_hidden = hidden_proj[label_mask]  # (num_label_tokens, D)
            label_ids = lmask[label_mask]  # (num_label_tokens,)
            label_batch_indices = label_positions[:, 0]  # (num_label_tokens,)

            # Create unique identifier for each (batch, label_id) pair
            max_label_id = lmask.max().item()
            unique_label_keys = (
                label_batch_indices * (max_label_id + 1) + label_ids
            )  # (num_label_tokens,)
            unique_keys, inverse_indices = torch.unique(
                unique_label_keys, return_inverse=True
            )

            # Compute label representations using scatter operations
            num_unique_labels = len(unique_keys)

            if cls_attn_weights is not None:
                # Attention-weighted averaging
                attn_weights = cls_attn_weights[label_mask]  # (num_label_tokens,)

                # Sum attention weights for normalization
                attn_sums = torch.zeros(num_unique_labels, device=hidden_states.device)
                attn_sums.scatter_add_(0, inverse_indices, attn_weights)
                attn_sums = attn_sums.clamp_min(1e-8)  # Avoid division by zero

                # Compute weighted sum of hidden states
                weighted_hidden = label_hidden * attn_weights.unsqueeze(
                    -1
                )  # (num_label_tokens, D)
                label_representations = torch.zeros(
                    num_unique_labels, projected_dim, device=hidden_states.device
                )
                label_representations.scatter_add_(
                    0,
                    inverse_indices.unsqueeze(-1).expand(-1, projected_dim),
                    weighted_hidden,
                )

                # Normalize by attention sums
                label_representations = label_representations / attn_sums.unsqueeze(-1)
            else:
                # Simple averaging
                # Count tokens per unique label
                counts = torch.zeros(num_unique_labels, device=hidden_states.device)
                counts.scatter_add_(
                    0,
                    inverse_indices,
                    torch.ones_like(inverse_indices, dtype=torch.float),
                )

                # Sum hidden states
                label_representations = torch.zeros(
                    num_unique_labels, projected_dim, device=hidden_states.device
                )
                label_representations.scatter_add_(
                    0,
                    inverse_indices.unsqueeze(-1).expand(-1, projected_dim),
                    label_hidden,
                )

                # Average
                label_representations = label_representations / counts.unsqueeze(
                    -1
                ).clamp_min(1)

            # Extract batch indices for each unique label
            label_batch_ids = unique_keys // (max_label_id + 1)  # (num_unique_labels,)

            # Get corresponding CLS tokens
            cls_for_labels = cls_tokens[label_batch_ids]  # (num_unique_labels, D)

            # Compute all similarities at once - SINGLE CALL TO compute_similarity!
            all_logits = self.compute_similarity(
                cls_for_labels, label_representations
            )  # (num_unique_labels, 1)

            return all_logits

        def _compute_loss(
            self, outputs_logits: torch.Tensor, labels: torch.Tensor
        ) -> Optional[torch.Tensor]:
            # Filter out padding values (-100) from labels
            temperature = 10 / len(labels)
            valid_mask = labels != -100
            valid_labels = labels[valid_mask].float().view(-1, 1)

            # Original BCE loss
            loss_values: torch.Tensor = self.loss_fn(outputs_logits, valid_labels)
            bce_loss = loss_values.mean() * self.scale_loss

            # Contrastive loss component - using torch.split
            valid_labels_flat = valid_labels.squeeze(-1)
            sigmoid_logits = torch.sigmoid(outputs_logits.squeeze(-1)).clamp(
                min=1e-6, max=1 - 1e-6
            )

            # Get split sizes for each batch (number of valid labels per sample)
            split_sizes = valid_mask.sum(dim=1).tolist()
            logits_splits = torch.split(sigmoid_logits, split_sizes)
            labels_splits = torch.split(valid_labels_flat, split_sizes)

            contrastive_losses = [
                self._compute_contrastive_loss(batch_logits, batch_labels)
                for batch_logits, batch_labels in zip(logits_splits, labels_splits)
            ]
            total_contrastive_loss = sum(contrastive_losses) * temperature

            return bce_loss + total_contrastive_loss

        def _compute_contrastive_loss(
            self, logits: torch.Tensor, labels: torch.Tensor
        ) -> torch.Tensor:
            # Separate positive (1) and negative (not 1) labels
            positive_mask = labels == 1.0
            negative_mask = labels != 1.0

            # Compute averages
            if positive_mask.any():
                avg_positive = logits[positive_mask].mean()
            else:
                avg_positive = logits.new_tensor(1.0)

            if negative_mask.any():
                avg_negative = logits[negative_mask].mean()
            else:
                avg_negative = logits.new_tensor(0.0)

            # Contrastive loss: encourage positive high, negative low
            return 1 - avg_positive + avg_negative  # avg_positive - avg_negative == 1

        @torch.inference_mode()
        def predict(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            lmask: torch.Tensor,
            activation_fn: Optional[str] = "sigmoid",
        ) -> List[List[float]]:
            self.eval()
            outputs = self.forward(
                input_ids=input_ids, attention_mask=attention_mask, lmask=lmask
            )
            logits_tensor = (
                outputs.logits if isinstance(outputs, GliZNetOutput) else outputs[0]
            )

            # Split logits based on number of labels per sample
            batch_size = lmask.shape[0]
            results = []
            logit_idx = 0

            for batch_idx in range(batch_size):
                sample_lmask = lmask[batch_idx]
                num_labels = len(
                    torch.unique(sample_lmask)[torch.unique(sample_lmask) > 0]
                )

                if num_labels > 0:
                    sample_logits = logits_tensor[logit_idx : logit_idx + num_labels]
                    logit_idx += num_labels

                    activate = (
                        torch.sigmoid
                        if activation_fn == "sigmoid"
                        else lambda x: torch.softmax(x, dim=0)
                    )
                    results.append(activate(sample_logits.squeeze(-1)).cpu().tolist())
                else:
                    results.append([])

            return results

    return GliZNetForSequenceClassification


GliZNetForSequenceClassification = create_gli_znet_for_sequence_classification()
