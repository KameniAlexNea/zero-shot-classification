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

        if config.scoring_method == "cosine":
            self.scoring = CosineScoring()
        else:
            self.scoring = BilinearScoring(hidden_size)
        self.dropout = nn.Dropout(config.dropout_rate)

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
        D = hidden_states.shape[-1]
        scores = torch.bmm(dense_labels, hidden_states.transpose(1, 2)) / (D**0.5)
        scores = scores.masked_fill(~text_mask.unsqueeze(1), float("-inf"))
        attn = F.softmax(scores, dim=2)
        agg_text_dense = torch.bmm(attn, hidden_states)  # (B, max_label_id, D)
        aggregated_text = agg_text_dense[all_batch_ids, all_label_ids - 1]  # (N, D)
        return aggregated_text, aggregated_labels

    def aggregate_labels(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        batch_size, _, _ = hidden_states.shape
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


class CLSLabelAttentionAggregator(LabelAggregator):
    """Extends LabelAggregator with a single-head self-attention layer over the
    [CLS, LAB_1, …, LAB_K] tokens.

    After the backbone the CLS and LAB tokens have never explicitly interacted
    as a group.  This lightweight module lets each LAB token attend to the CLS
    summary (and to the other labels) and refines CLS with all label context,
    pulling them into a shared subspace before scoring — especially helpful for
    cosine similarity.

    Architecture (per sample):
        tokens  = [CLS_h, LAB_1_h, …, LAB_K_h]   shape (1+K, D)
        tokens' = tokens + SelfAttn(LayerNorm(tokens))
        text_repr   = tokens'[:, 0]   (refined CLS)
        label_repr  = tokens'[:, 1:]  (refined LAB embeddings)
    """

    def __init__(self, config: GliZNetConfig):
        super().__init__(config)
        D = config.backbone_config.hidden_size
        self.cls_lab_attn = nn.MultiheadAttention(
            embed_dim=D, num_heads=1, batch_first=True, dropout=config.dropout_rate
        )
        self.norm = nn.LayerNorm(D)

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
        """Single-head self-attention over [CLS, LAB_1…LAB_K] per sample.

        Returns:
            aggregated_text: (N, D) — refined CLS per label slot.
            aggregated_labels: (N, D) — refined LAB embeddings.
        """
        B, _, D = hidden_states.shape

        # CLS token: position 0
        cls_h = hidden_states[:, 0:1, :]  # (B, 1, D)
        lab_h = dense_labels  # (B, K, D)

        # Sequence: [CLS, LAB_1, …, LAB_K]  →  shape (B, 1+K, D)
        seq = torch.cat([cls_h, lab_h], dim=1)

        # Key/value padding mask: True = ignore.
        # Slot i+1 (0-indexed) is padding if no label occupies it.
        lab_counts = dense_labels.abs().sum(-1) != 0  # (B, K) bool — True = valid
        cls_valid = torch.ones(B, 1, dtype=torch.bool, device=hidden_states.device)
        key_padding_mask = ~torch.cat([cls_valid, lab_counts], dim=1)  # (B, 1+K)

        normed = self.norm(seq)
        attn_out, _ = self.cls_lab_attn(
            normed, normed, normed, key_padding_mask=key_padding_mask
        )
        seq = seq + attn_out  # residual

        # Refined CLS and LABs
        cls_refined = seq[:, 0, :]  # (B, D)
        lab_refined = seq[:, 1:, :]  # (B, K, D)

        # Broadcast refined CLS to every valid label slot of that sample
        aggregated_text = cls_refined[all_batch_ids]  # (N, D)
        aggregated_labels_out = lab_refined[all_batch_ids, all_label_ids - 1]  # (N, D)

        return aggregated_text, aggregated_labels_out
