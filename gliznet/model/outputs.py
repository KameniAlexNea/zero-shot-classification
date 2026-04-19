from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import ModelOutput


@dataclass
class GliZNetOutput(ModelOutput):
    """Output class for GliZNet model.

    Args:
        loss: Training loss (optional)
        logits: Classification logits, shape (N, 1) — one score per label span
        batch_indices: Batch index for each label span (N,)
        label_ids: Label ID for each label span (N,)
        label_embeddings: Projected label embeddings (N, D)
        text_embeddings: Label-specific attended text representations (N, D).
                         One row per label span; NOT the CLS token.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.Tensor] = None
    batch_indices: Optional[torch.Tensor] = None
    label_ids: Optional[torch.Tensor] = None
    label_embeddings: Optional[torch.Tensor] = None
    text_embeddings: Optional[torch.Tensor] = None  # label-specific attended text (one row per label span)
