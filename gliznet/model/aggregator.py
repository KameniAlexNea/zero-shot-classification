from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from gliznet.model.config import GliZNetConfig
from gliznet.model.similarity import SimilarityHead


class LabelAggregator(nn.Module):
    """Aggregates label token embeddings and computes similarities using token-level attention."""

    def __init__(self, config: GliZNetConfig):
        super().__init__()
        self.config = config

        hidden_size = config.backbone_config.hidden_size
        projected_dim = config.projected_dim or hidden_size

        self.projected_dim = projected_dim
        self.text_projector = self._build_projector(hidden_size, projected_dim)
        self.label_projector = self._build_projector(hidden_size, projected_dim)
        self.similarity_head = SimilarityHead(config, projected_dim)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Temperature for attention softmax (learnable)
        self.attention_temperature = nn.Parameter(torch.tensor(1.0))

    def _build_projector(self, input_dim: int, output_dim: int) -> nn.Module:
        if input_dim == output_dim and not self.config.use_projection_layernorm:
            return nn.Identity()
        layers = [nn.Linear(input_dim, output_dim)]
        if self.config.use_projection_layernorm:
            layers.append(nn.LayerNorm(output_dim))
        return nn.Sequential(*layers)

    def aggregate_labels(
        self,
        input_ids: torch.Tensor,
        lmask: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        if self.config.use_lab_token_for_labels:
            # Mode: Use [LAB] token embeddings as label representations
            lab_mask = input_ids == self.config.lab_token_id
            if not lab_mask.any():
                empty_idx = torch.empty(0, dtype=torch.long, device=device)
                empty_emb = torch.empty(0, self.projected_dim, device=device)
                return empty_emb, empty_idx, empty_idx

            label_hidden = self.dropout(self.label_projector(hidden_states[lab_mask]))

            batch_indices_all = (
                torch.arange(batch_size, device=device)
                .unsqueeze(1)
                .expand(batch_size, seq_len)
            )
            all_batch_ids = batch_indices_all[lab_mask]

            lab_counts = lab_mask.sum(dim=1)
            max_labels = self.config.max_labels

            batch_label_grid = (
                torch.arange(batch_size, device=device)
                .unsqueeze(1)
                .expand(batch_size, max_labels)
            )
            label_id_grid = (
                torch.arange(1, max_labels + 1, device=device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

            valid_mask = label_id_grid <= lab_counts.unsqueeze(1)
            all_batch_ids = batch_label_grid.reshape(-1)[valid_mask.reshape(-1)]
            all_label_ids = label_id_grid.reshape(-1)[valid_mask.reshape(-1)]
            aggregated_labels = label_hidden

        else:
            # Mode: Average label token embeddings
            label_mask = lmask > 0
            if not label_mask.any():
                empty_idx = torch.empty(0, dtype=torch.long, device=device)
                empty_emb = torch.empty(0, self.projected_dim, device=device)
                return empty_emb, empty_idx, empty_idx

            label_hidden = self.dropout(self.label_projector(hidden_states[label_mask]))

            batch_indices_all = (
                torch.arange(batch_size, device=device)
                .unsqueeze(1)
                .expand(batch_size, seq_len)
            )
            token_batch_ids = batch_indices_all[label_mask]
            token_label_ids = lmask[label_mask].long()

            max_label_id = self.config.max_labels
            num_slots = batch_size * max_label_id
            flat_indices = token_batch_ids * max_label_id + (token_label_ids - 1)

            projected_dim = label_hidden.shape[-1]
            aggregated = torch.zeros(
                num_slots, projected_dim, device=device, dtype=label_hidden.dtype
            )
            counts = torch.zeros(num_slots, device=device, dtype=label_hidden.dtype)

            aggregated.index_add_(0, flat_indices, label_hidden)
            counts.index_add_(
                0,
                flat_indices,
                torch.ones(len(flat_indices), device=device, dtype=label_hidden.dtype),
            )

            valid_mask = counts > 0
            if not valid_mask.any():
                empty_idx = torch.empty(0, dtype=torch.long, device=device)
                empty_emb = torch.empty(0, projected_dim, device=device)
                return empty_emb, empty_idx, empty_idx

            aggregated_labels = aggregated[valid_mask] / counts[valid_mask].unsqueeze(
                -1
            )

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

        return aggregated_labels, all_batch_ids, all_label_ids

    def forward(
        self,
        hidden_states: torch.Tensor,
        lmask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Aggregate label representations and compute similarities using token-level attention.

        Args:
            hidden_states: Encoder outputs (B, L, H)
            lmask: Label mask where >0 indicates label tokens (B, L)
            input_ids: Input token IDs (B, L)
            attention_mask: Attention mask for text tokens (B, L)

        Returns:
            logits: Similarity scores (N, 1)
            batch_indices: Batch index for each score (N,)
            label_ids: Label ID for each score (N,)
            label_embeddings: Aggregated label embeddings (N, D)
            logit_scale: Current temperature scale
            text_aggregations: Label-specific text representations (N, D)
        """
        # Project ALL tokens (not just CLS)
        projected_all = self.dropout(self.text_projector(hidden_states))  # (B, L, D)

        # Identify text token positions (exclude label/special tokens)
        if self.config.use_lab_token_for_labels:
            lab_token_mask = input_ids == self.config.lab_token_id
            text_mask = (lmask == 0) & (attention_mask == 1) & (~lab_token_mask)
        else:
            text_mask = (lmask == 0) & (attention_mask == 1)

        aggregated_labels, all_batch_ids, all_label_ids = self.aggregate_labels(
            input_ids, lmask, hidden_states
        )

        # Early return if no label spans were found (e.g. malformed tokenization).
        if aggregated_labels.shape[0] == 0:
            logit_scale = self.similarity_head.logit_scale.clamp(-10, 10)
            empty_logits = torch.empty(0, 1, device=hidden_states.device)
            return (
                empty_logits,
                all_batch_ids,
                all_label_ids,
                aggregated_labels,
                logit_scale,
                aggregated_labels,
            )

        # Token-level attention: attend over the text tokens of each sample per label.
        # Dense batched matmul avoids the (N, L, D) intermediate tensor (N = total label spans).
        # Pack labels into (B, max_labels, D) and use two batched bmm ops instead.
        B = hidden_states.shape[0]
        D = aggregated_labels.shape[-1]
        max_label_id = self.config.max_labels
        scale = self.attention_temperature.abs().clamp(min=0.1) * (D ** 0.5)

        dense_labels = aggregated_labels.new_zeros(B, max_label_id, D)
        dense_labels[all_batch_ids, all_label_ids - 1] = aggregated_labels  # scatter

        # (B, max_labels, D) @ (B, D, L) → (B, max_labels, L)
        scores_dense = torch.bmm(dense_labels, projected_all.transpose(1, 2)) / scale
        # Mask non-text positions: text_mask (B, L) broadcast to (B, 1, L)
        scores_dense = scores_dense.masked_fill(~text_mask.unsqueeze(1), float("-inf"))
        attn_weights_dense = F.softmax(scores_dense, dim=2)  # (B, max_labels, L)

        # (B, max_labels, L) @ (B, L, D) → (B, max_labels, D)
        aggregated_text_dense = torch.bmm(attn_weights_dense, projected_all)

        # Extract only the valid (N,) label slots
        aggregated_text = aggregated_text_dense[all_batch_ids, all_label_ids - 1]  # (N, D)

        logits, logit_scale = self.similarity_head(aggregated_text, aggregated_labels)

        return (
            logits,
            all_batch_ids,
            all_label_ids,
            aggregated_labels,
            logit_scale,
            aggregated_text,
        )
