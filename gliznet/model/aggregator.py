import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from gliznet.model.config import GliZNetConfig


class BilinearScoring(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.bilinear = nn.Bilinear(hidden_size, hidden_size, 1)

    def forward(self, text: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.bilinear(text, labels)


class CosineScoring(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(self, text: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        text_norm = F.normalize(text, p=2, dim=-1)
        label_norm = F.normalize(labels, p=2, dim=-1)
        scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        return (text_norm * label_norm).sum(dim=-1, keepdim=True) * scale


class LabelAggregator(nn.Module):
    """Aggregates label token embeddings and computes similarities using token-level attention."""

    def __init__(self, config: GliZNetConfig):
        super().__init__()
        self.config = config

        hidden_size = config.backbone_config.hidden_size
        self.hidden_size = hidden_size
        self.max_labels = config.max_labels
        self.lab_token_id = config.lab_token_id
        self._inv_scale = hidden_size**-0.5

        if config.scoring_method == "cosine":
            self.scoring = CosineScoring()
        else:
            self.scoring = BilinearScoring(hidden_size)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Pre-register grid buffers (created once, never recomputed)
        self.register_buffer(
            "_label_id_grid",
            torch.arange(1, config.max_labels + 1).unsqueeze(0),
            persistent=False,
        )
        self.register_buffer(
            "_batch_label_grid_template",
            torch.zeros(1, config.max_labels, dtype=torch.long),
            persistent=False,
        )

    def _text_repr(
        self,
        dense_labels: torch.Tensor,
        aggregated_labels: torch.Tensor,
        hidden_states: torch.Tensor,
        text_mask: torch.Tensor,
        all_batch_ids: torch.Tensor,
        all_label_ids: torch.Tensor,
        max_label_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cross-attention of each label over text tokens.

        Returns:
            aggregated_text: (N, D) label-specific text representations.
            aggregated_labels: (N, D) label embeddings (unchanged in base class).
        """
        # (B, K, L) = (B, K, D) @ (B, D, L) scaled
        scores = torch.bmm(dense_labels, hidden_states.transpose(1, 2))
        scores = scores * self._inv_scale
        scores.masked_fill_(~text_mask.unsqueeze(1), float("-inf"))
        attn = F.softmax(scores, dim=2)
        agg_text_dense = torch.bmm(attn, hidden_states)  # (B, K, D)
        aggregated_text = agg_text_dense[all_batch_ids, all_label_ids - 1]  # (N, D)
        return aggregated_text, aggregated_labels

    def aggregate_labels(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        batch_size = hidden_states.shape[0]
        device = hidden_states.device

        lab_mask = input_ids == self.lab_token_id
        if not lab_mask.any():
            empty_idx = torch.empty(0, dtype=torch.long, device=device)
            empty_emb = torch.empty(0, self.hidden_size, device=device)
            return empty_emb, empty_idx, empty_idx

        label_hidden = self.dropout(hidden_states[lab_mask])

        lab_counts = lab_mask.sum(dim=1)

        # Use pre-registered grids (expand is free — no copy)
        label_id_grid = self._label_id_grid.expand(batch_size, -1)
        batch_label_grid = (
            torch.arange(batch_size, device=device)
            .unsqueeze(1)
            .expand(batch_size, self.max_labels)
        )

        valid_mask = label_id_grid <= lab_counts.unsqueeze(1)
        flat_valid = valid_mask.reshape(-1)
        all_batch_ids = batch_label_grid.reshape(-1)[flat_valid]
        all_label_ids = label_id_grid.reshape(-1)[flat_valid]

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
        lab_token_mask = input_ids == self.lab_token_id
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
        max_label_id = self.max_labels

        dense_labels = aggregated_labels.new_zeros(B, max_label_id, D)
        dense_labels[all_batch_ids, all_label_ids - 1] = aggregated_labels

        aggregated_text, aggregated_labels = self._text_repr(
            dense_labels,
            aggregated_labels,
            hidden_states,
            text_mask,
            all_batch_ids,
            all_label_ids,
            max_label_id,
        )

        logits = self.scoring(aggregated_text, aggregated_labels)

        return (
            logits,
            all_batch_ids,
            all_label_ids,
            aggregated_labels,
            aggregated_text,
        )

