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


class LabelContextAttention(nn.Module):
    """Cooperative label enrichment: each label's representation is fused with its
    first-pass text evidence, then labels attend to each other. This lets label_i
    see what text evidence label_j found, enabling cooperative routing."""

    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.fuse = nn.Linear(hidden_size * 2, hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

    def forward(
        self,
        dense_labels: torch.Tensor,  # (B, K, D) label embeddings
        dense_text: torch.Tensor,  # (B, K, D) first-pass text pooling per label
        label_mask: torch.Tensor,  # (B, K) bool, True = valid label
    ) -> torch.Tensor:  # (B, K, D) enriched label embeddings
        # Fuse each label with its text evidence to create the context memory
        fused = self.fuse(torch.cat([dense_labels, dense_text], dim=-1))
        # Cross-attention: labels query the fused peer context
        pad_mask = ~label_mask  # True = ignore
        out, _ = self.attn(
            query=dense_labels, key=fused, value=fused, key_padding_mask=pad_mask
        )
        return F.normalize(dense_labels + out, p=2, dim=-1)


class LabelAggregator(nn.Module):
    """Aggregates label token embeddings and computes similarities using token-level attention."""

    def __init__(self, config: GliZNetConfig):
        super().__init__()
        hidden_size = config.backbone_config.hidden_size
        self.hidden_size = hidden_size
        self.lab_token_id = config.lab_token_id
        # Learned temperature for attention over unit-norm vectors
        self.attn_temperature = nn.Parameter(
            torch.tensor(math.log(math.sqrt(float(hidden_size))))
        )

        if config.scoring_method == "cosine":
            self.scoring = CosineScoring()
        else:
            self.scoring = BilinearScoring(hidden_size)
        self.dropout = nn.Dropout(config.dropout_rate)
        if config.enrich_labels:
            self.label_context = LabelContextAttention(hidden_size)
        else:
            self.label_context = None

    def _text_repr_dense(
        self,
        dense_labels: torch.Tensor,  # (B, K, D)
        hidden_states: torch.Tensor,  # (B, L, D)
        text_mask: torch.Tensor,  # (B, L)
    ) -> torch.Tensor:  # (B, K, D)
        """First-pass text pooling returning the full dense (B, K, D) tensor."""
        scale = self.attn_temperature.exp()
        scores = torch.bmm(dense_labels, hidden_states.transpose(1, 2)) * scale
        scores.masked_fill_(~text_mask.unsqueeze(1), float("-inf"))
        attn = F.softmax(scores, dim=2)
        return torch.bmm(attn, hidden_states)  # (B, K, D)

    def _text_repr(
        self,
        dense_labels: torch.Tensor,
        hidden_states: torch.Tensor,
        text_mask: torch.Tensor,
        all_batch_ids: torch.Tensor,
        all_label_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-attention of each label over text tokens.

        Returns:
            aggregated_text: (N, D) label-specific text representations.
        """
        agg_text_dense = self._text_repr_dense(
            dense_labels, hidden_states, text_mask
        )  # (B, K, D)
        aggregated_text = agg_text_dense[all_batch_ids, all_label_ids - 1]  # (N, D)
        return aggregated_text

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
            return empty_emb, empty_idx, empty_idx, 0

        label_hidden = self.dropout(hidden_states[lab_mask])

        lab_counts = lab_mask.sum(dim=1)
        max_k = int(lab_counts.max().item())

        label_id_grid = (
            torch.arange(1, max_k + 1, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        batch_label_grid = (
            torch.arange(batch_size, device=device)
            .unsqueeze(1)
            .expand(batch_size, max_k)
        )

        valid_mask = label_id_grid <= lab_counts.unsqueeze(1)
        flat_valid = valid_mask.reshape(-1)
        all_batch_ids = batch_label_grid.reshape(-1)[flat_valid]
        all_label_ids = label_id_grid.reshape(-1)[flat_valid]

        return label_hidden, all_batch_ids, all_label_ids, max_k

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
            text_aggregations: Label-specific text representations (N, D)
        """
        B, L, D = hidden_states.shape

        hidden_states = F.normalize(hidden_states, p=2, dim=-1)

        # Identify text token positions (exclude label/special tokens)
        lab_token_mask = input_ids == self.lab_token_id
        text_mask = (lmask == 0) & (attention_mask == 1) & (~lab_token_mask)

        aggregated_labels, all_batch_ids, all_label_ids, max_k = self.aggregate_labels(
            input_ids, hidden_states
        )

        # Early return if no label spans were found
        if max_k == 0:
            empty_logits = torch.empty(0, 1, device=hidden_states.device)
            return (
                empty_logits,
                all_batch_ids,
                all_label_ids,
                aggregated_labels,
                aggregated_labels,
            )

        # Token-level attention: attend over text tokens per label.
        max_label_id = max_k

        dense_labels = aggregated_labels.new_zeros(B, max_label_id, D)
        dense_labels[all_batch_ids, all_label_ids - 1] = aggregated_labels

        # Let all labels interact with each other and the global CLS token
        if self.label_context is not None:
            label_mask = torch.zeros(
                B, max_label_id, dtype=torch.bool, device=hidden_states.device
            )
            label_mask[all_batch_ids, all_label_ids - 1] = True
            # Round 1: independent text pooling to get initial text locations
            dense_text = self._text_repr_dense(dense_labels, hidden_states, text_mask)
            # Cooperative enrichment: each label attends to peers + their text locations
            dense_labels = self.label_context(dense_labels, dense_text, label_mask)
            # Gather enriched label embeddings
            aggregated_labels = dense_labels[all_batch_ids, all_label_ids - 1]

        aggregated_text = self._text_repr(
            dense_labels,
            hidden_states,
            text_mask,
            all_batch_ids,
            all_label_ids,
        )

        logits = self.scoring(aggregated_text, aggregated_labels)

        return (
            logits,
            all_batch_ids,
            all_label_ids,
            aggregated_labels,
            aggregated_text,
        )
