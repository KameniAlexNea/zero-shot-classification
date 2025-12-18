from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModel, BertConfig, BertPreTrainedModel
from transformers.modeling_outputs import ModelOutput


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
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")
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
            loss = None
            if labels is not None:
                loss = self._compute_loss(outputs_logits, labels)
                # Add Barlow regularization if weight > 0
                if self.config.barlow_loss_weight > 0:
                    loss = (
                        loss
                        + self.config.barlow_loss_weight
                        * self._compute_barlow_loss(outputs_logits)
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

        def _compute_batch_logits(
            self,
            hidden_states: torch.Tensor,  # (B, L, H)
            lmask: torch.Tensor,  # (B, L), values: 0 = text, 1... = label groups
            cls_attn_weights: Optional[torch.Tensor],  # (B, L)
        ) -> torch.Tensor:
            """
            Compute logits between [CLS] and label representations per sample.

            If use_separator_pooling=True, uses separator token embeddings directly.
            Otherwise, uses averaged label token embeddings (default implementation).
            """
            # Check if we should use separator token pooling approach
            if getattr(self.config, "use_separator_pooling", False):
                return self._compute_batch_logits_with_separator_tokens(
                    hidden_states, lmask, cls_attn_weights
                )
            return self._compute_batch_logits_original(
                hidden_states, lmask, cls_attn_weights
            )

        def _compute_batch_logits_with_separator_tokens(
            self,
            hidden_states: torch.Tensor,  # (B, L, H)
            lmask: torch.Tensor,  # (B, L), values: 0 = text, 1... = label groups
            cls_attn_weights: Optional[torch.Tensor],  # (B, L)
        ) -> torch.Tensor:
            """
            Optimized version that uses separator token embeddings directly.
            This is used when custom tokens (like [LAB]) have been added to the model.
            """
            # Project CLS tokens and separator tokens separately
            cls_tokens = self.cls_proj(hidden_states[:, 0])  # (B, D)

            # Create mask for separator tokens (lmask == 0, but exclude CLS position)
            separator_mask = lmask.bool()

            # Get separator token positions and their embeddings
            separator_positions = (
                separator_mask.nonzero()
            )  # (num_separators, 2) - [batch_idx, seq_idx]

            separator_hidden = self.label_proj(
                hidden_states[separator_mask]
            )  # (num_separators, D)
            separator_batch_indices = separator_positions[:, 0]  # (num_separators,)

            # Get corresponding CLS tokens for each separator
            cls_for_separators = cls_tokens[
                separator_batch_indices
            ]  # (num_separators, D)

            # Compute similarities between CLS and separator tokens
            all_logits = self.compute_similarity(
                cls_for_separators, separator_hidden
            )  # (num_separators, 1)

            return all_logits

        def _compute_batch_logits_original(
            self,
            hidden_states: torch.Tensor,  # (B, L, H)
            lmask: torch.Tensor,  # (B, L), values: 0 = text, 1... = label groups
            cls_attn_weights: Optional[torch.Tensor],  # (B, L)
        ) -> torch.Tensor:
            """
            Original implementation: Compute logits between [CLS] and averaged label embeddings per sample.
            Fully vectorized implementation - computes similarity only once!
            """
            # Project CLS tokens and all tokens separately
            cls_tokens = self.cls_proj(hidden_states[:, 0])  # (B, D)
            projected_dim = cls_tokens.shape[-1]

            # Create mask for non-text tokens (label tokens have lmask > 0)
            label_mask = lmask > 0  # (B, L)

            # Get all label tokens and their metadata
            label_positions = (
                label_mask.nonzero()
            )  # (num_label_tokens, 2) - [batch_idx, seq_idx]
            label_hidden = self.label_proj(
                hidden_states[label_mask]
            )  # (num_label_tokens, D)
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
            """Compute composite loss with BCE, contrastive, and optional Barlow components.

            Args:
                outputs_logits: Model predictions (num_labels, 1)
                labels: Ground truth labels (batch_size, max_labels) with -100 for padding

            Returns:
                Combined loss tensor
            """
            # Filter out padding values (-100) from labels
            valid_mask = labels != -100
            num_valid_labels = valid_mask.sum().item()

            # Fix: temperature should scale with total number of valid labels, not batch size
            temperature = self.config.temperature * (10.0 / max(num_valid_labels, 1))

            valid_labels = labels[valid_mask].float().view(-1, 1)

            # BCE loss component
            loss_values: torch.Tensor = self.loss_fn(outputs_logits, valid_labels)
            bce_loss = loss_values.mean() * self.scale_loss

            # Contrastive loss component (hard negative mining)
            valid_labels_flat = valid_labels.squeeze(-1)
            sigmoid_logits = torch.sigmoid(outputs_logits.squeeze(-1)).clamp(
                min=1e-6, max=1 - 1e-6
            )

            # Get split sizes for each batch (number of valid labels per sample)
            split_sizes = valid_mask.sum(dim=1).tolist()
            logits_splits = torch.split(sigmoid_logits, split_sizes)
            labels_splits = torch.split(valid_labels_flat, split_sizes)

            contrastive_loss = (
                sum(map(self._compute_contrastive_loss, logits_splits, labels_splits))
                * temperature
                * self.config.contrastive_loss_weight
            )

            return bce_loss + contrastive_loss

        def _compute_barlow_loss(
            self, logits: torch.Tensor, coef: float = 0.005
        ) -> torch.Tensor:
            """
            Compute contrastive loss for a single batch of logits and labels.
            This is a simplified version that handles the case where there are no valid comparisons.
            """
            if logits.numel() == 0:
                return torch.tensor(0.0, device=logits.device)
            # Barlow Twins loss implementation
            # Normalize logits
            z = logits.unsqueeze(0) if logits.dim() == 1 else logits
            z_norm = (z - z.mean(dim=0)) / (z.std(dim=0) + 1e-6)

            # Compute cross-correlation matrix
            c = torch.mm(z_norm.T, z_norm) / z_norm.shape[0]

            # Barlow Twins loss: on-diagonal should be close to 1, off-diagonal close to 0
            off_diag = c.flatten()[1:].view(c.size(0), -1).pow_(2).sum()
            return coef * off_diag

        def _compute_contrastive_loss(
            self, logits: torch.Tensor, labels: torch.Tensor
        ) -> torch.Tensor:
            positive_logits = logits[labels == 1.0]
            negative_logits = logits[labels != 1.0]

            # If no valid comparisons, skip
            if positive_logits.numel() == 0 or negative_logits.numel() == 0:
                return torch.tensor(0.0, device=logits.device)

            # Get hardest cases
            min_pos = positive_logits.min()
            max_neg = negative_logits.max()

            # Loss: we want min(pos) > max(neg)
            return F.relu(self.margin + max_neg - min_pos)

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
                # More efficient: use max instead of unique
                num_labels = sample_lmask.max().item()

                if num_labels > 0:
                    sample_logits = logits_tensor[logit_idx : logit_idx + num_labels]
                    logit_idx += num_labels

                    # Apply activation function
                    if activation_fn == "sigmoid":
                        scores = torch.sigmoid(sample_logits.squeeze(-1))
                    else:
                        scores = torch.softmax(sample_logits.squeeze(-1), dim=0)

                    results.append(scores.cpu().tolist())
                else:
                    results.append([])

            return results

    return GliZNetForSequenceClassification


GliZNetForSequenceClassification = create_gli_znet_for_sequence_classification()
