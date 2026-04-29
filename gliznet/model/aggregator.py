from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from gliznet.model.config import GliZNetConfig


class LabelAggregator(nn.Module):
    """Aggregates label token embeddings and computes similarities using token-level attention."""

    def __init__(self, config: GliZNetConfig):
        super().__init__()
        self.config = config

        hidden_size = config.backbone_config.hidden_size
        self.hidden_size = hidden_size

        self.bilinear = nn.Bilinear(hidden_size, hidden_size, 1)
        self.dropout = nn.Dropout(config.dropout_rate)

    def aggregate_labels(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        lab_mask = input_ids == self.config.lab_token_id
        if not lab_mask.any():
            empty_idx = torch.empty(0, dtype=torch.long, device=device)
            empty_emb = torch.empty(0, self.hidden_size, device=device)
            return empty_emb, empty_idx, empty_idx

        label_hidden = self.dropout(hidden_states[lab_mask])

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

        return label_hidden, all_batch_ids, all_label_ids

    def forward(
        self,
        hidden_states: torch.Tensor,
        lmask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        inference: bool = False,
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
            text_aggregations: Label-specific text representations (N, D)
        """
        B, L, D = hidden_states.shape

        # Identify text token positions (exclude label/special tokens)
        lab_token_mask = input_ids == self.config.lab_token_id
        text_mask = (lmask == 0) & (attention_mask == 1) & (~lab_token_mask)

        aggregated_labels, all_batch_ids, all_label_ids = self.aggregate_labels(
            input_ids, hidden_states
        )

        # Early return if no label spans were found
        if aggregated_labels.shape[0] == 0:
            empty_logits = torch.empty(0, 1, device=hidden_states.device)
            return (
                empty_logits,
                all_batch_ids,
                all_label_ids,
                aggregated_labels,
                aggregated_labels,
            )

        # Token-level attention: attend over text tokens per label.
        max_label_id = (
            int(all_label_ids.max().item()) if inference else self.config.max_labels
        )
        scale = D**0.5

        dense_labels = aggregated_labels.new_zeros(B, max_label_id, D)
        dense_labels[all_batch_ids, all_label_ids - 1] = aggregated_labels

        # (B, max_labels, D) @ (B, D, L) → (B, max_labels, L)
        scores_dense = torch.bmm(dense_labels, hidden_states.transpose(1, 2)) / scale
        scores_dense = scores_dense.masked_fill(~text_mask.unsqueeze(1), float("-inf"))
        attn_weights_dense = F.softmax(scores_dense, dim=2)  # (B, max_labels, L)

        # (B, max_labels, L) @ (B, L, D) → (B, max_labels, D)
        aggregated_text_dense = torch.bmm(attn_weights_dense, hidden_states)

        # Extract only the valid (N,) label slots
        aggregated_text = aggregated_text_dense[
            all_batch_ids, all_label_ids - 1
        ]  # (N, D)

        logits = self.bilinear(aggregated_text, aggregated_labels)

        return (
            logits,
            all_batch_ids,
            all_label_ids,
            aggregated_labels,
            aggregated_text,
        )
