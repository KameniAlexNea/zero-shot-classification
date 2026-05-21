from typing import Tuple

import torch
import torch.nn as nn

from gliznet.model.config import GliZNetConfig


class BilinearScoring(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.bilinear = nn.Bilinear(hidden_size, hidden_size, 1)

    def forward(self, text: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.bilinear(text, labels)


class LabelContextFusion(nn.Module):
    """Label interaction through fused text-label representations.

    Each label's identity is fused with CLS (shared text context), producing
    per-label "claims on text". Labels then attend to each other's fused views,
    enabling competitive/cooperative dynamics (e.g. "sports" suppresses "politics",
    "basketball" reinforces "NBA").

    The attention output serves as the label-specific text representation."""

    def __init__(self, hidden_size: int, dropout: float = 0.1, num_heads: int = 8):
        super().__init__()
        self.fuse = nn.Linear(hidden_size * 2, hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        dense_labels: torch.Tensor,  # (B, K, D) label embeddings
        text_cls: torch.Tensor,  # (B, D) CLS text representation
        label_mask: torch.Tensor,  # (B, K) bool, True = valid label
    ) -> torch.Tensor:  # (B, K, D) label-specific text representations
        # Expand CLS to match label positions
        text_expanded = text_cls.unsqueeze(1).expand_as(dense_labels)  # (B, K, D)
        # Fuse: each label's interpretation of shared text
        fused = self.fuse(torch.cat([text_expanded, dense_labels], dim=-1))
        # Labels query peer fused views
        pad_mask = ~label_mask  # True = ignore
        out, _ = self.attn(
            query=fused, key=fused, value=fused, key_padding_mask=pad_mask
        )
        return self.norm(out)


class LabelAggregator(nn.Module):
    """Aggregates label token embeddings and computes scores via label interaction."""

    def __init__(self, config: GliZNetConfig):
        super().__init__()
        self.config = config

        hidden_size = config.backbone_config.hidden_size
        self.hidden_size = hidden_size
        self.lab_token_id = config.lab_token_id

        self.scoring = BilinearScoring(hidden_size)
        if config.enrich_labels:
            self.context = LabelContextFusion(hidden_size, dropout=config.dropout_rate)
        else:
            self.context = None

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

        label_hidden = hidden_states[lab_mask]

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
        """Aggregate label representations and compute scores.

        The encoder already performed full self-attention between text and label
        tokens. CLS captures the global text representation; [LAB] tokens capture
        label-specific representations enriched by text context.

        When label context is enabled, labels attend to each other's fused
        (CLS + label) views. The attention output is the text representation.

        Returns:
            logits: Similarity scores (N, 1)
            batch_indices: Batch index for each score (N,)
            label_ids: Label ID for each score (N,)
            label_embeddings: Label embeddings (N, D)
            text_repr: Label-specific text representations (N, D)
        """
        B, _, D = hidden_states.shape

        aggregated_labels, all_batch_ids, all_label_ids, max_k = self.aggregate_labels(
            input_ids, hidden_states
        )

        # Early return if no labels found
        if max_k == 0:
            empty_logits = torch.empty(0, 1, device=hidden_states.device)
            return (
                empty_logits,
                all_batch_ids,
                all_label_ids,
                aggregated_labels,
                aggregated_labels,
            )

        if self.context is not None:
            # Build dense label tensor for attention
            dense_labels = aggregated_labels.new_zeros(B, max_k, D)
            dense_labels[all_batch_ids, all_label_ids - 1] = aggregated_labels

            label_mask = torch.zeros(
                B, max_k, dtype=torch.bool, device=hidden_states.device
            )
            label_mask[all_batch_ids, all_label_ids - 1] = True

            # CLS as shared text context
            text_cls = hidden_states[:, 0]  # (B, D)

            # Label interaction: attend to peer fused views
            text_repr_dense = self.context(dense_labels, text_cls, label_mask)

            # Gather per-label text representations
            text_repr = text_repr_dense[all_batch_ids, all_label_ids - 1]  # (N, D)
        else:
            # No interaction: use CLS directly
            text_repr = hidden_states[:, 0][all_batch_ids]  # (N, D)

        logits = self.scoring(text_repr, aggregated_labels)

        return (
            logits,
            all_batch_ids,
            all_label_ids,
            aggregated_labels,
            text_repr,
        )
