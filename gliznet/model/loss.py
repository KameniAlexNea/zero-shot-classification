import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from gliznet.model.config import GliZNetConfig

logger = logging.getLogger(__name__)


class SoftmaxLoss(nn.Module):
    """One-vs-negatives softmax loss with optional additive margin."""

    def __init__(self, config: GliZNetConfig):
        super().__init__()
        self.margin = config.supcon_margin

    def forward(
        self, dense_logits: torch.Tensor, labels: torch.Tensor, **_
    ) -> torch.Tensor:
        mask_valid = labels != -100
        targets_clean = labels.clone()
        targets_clean[~mask_valid] = 0.0

        pos_mask = targets_clean > 0.5
        has_positives = pos_mask.any(dim=1)
        if not has_positives.any():
            return torch.tensor(
                0.0,
                device=dense_logits.device,
                dtype=dense_logits.dtype,
                requires_grad=True,
            )

        logits = dense_logits[has_positives]
        pos_mask = pos_mask[has_positives]
        mask_valid = mask_valid[has_positives]
        neg_mask = mask_valid & ~pos_mask

        neg_logits = logits.masked_fill(~neg_mask, float("-inf"))
        if self.margin > 0.0:
            neg_logits = neg_logits + self.margin
        neg_lse = torch.logsumexp(neg_logits, dim=1)

        neg_lse_exp = neg_lse.unsqueeze(1).expand_as(logits)
        denom_lse = torch.logsumexp(torch.stack([logits, neg_lse_exp], dim=2), dim=2)
        per_pos_loss = denom_lse - logits

        valid_pos = pos_mask & torch.isfinite(logits)
        per_pos_loss = per_pos_loss.masked_fill(~valid_pos, 0.0)

        if torch.isnan(per_pos_loss).any() or torch.isinf(per_pos_loss).any():
            return torch.tensor(
                0.0,
                device=dense_logits.device,
                dtype=dense_logits.dtype,
                requires_grad=True,
            )

        num_pos = valid_pos.sum(dim=1).float().clamp(min=1e-6)
        return (per_pos_loss.sum(dim=1) / num_pos).mean()


class RepulsionLoss(nn.Module):
    """Penalize high cosine similarity between different labels in the same sample."""

    def __init__(self, config: GliZNetConfig):
        super().__init__()
        self.threshold = config.repulsion_threshold

    def forward(
        self,
        label_embeddings: torch.Tensor,
        label_ids: torch.Tensor,
        batch_indices: torch.Tensor,
        **_,
    ) -> torch.Tensor:
        if label_embeddings.numel() == 0:
            return torch.tensor(
                0.0,
                device=label_embeddings.device,
                dtype=label_embeddings.dtype,
                requires_grad=True,
            )

        embeddings_norm = F.normalize(label_embeddings, p=2, dim=-1)
        sim_matrix = torch.matmul(embeddings_norm, embeddings_norm.T)

        diff_label_mask = label_ids.unsqueeze(0) != label_ids.unsqueeze(1)
        same_batch_mask = batch_indices.unsqueeze(0) == batch_indices.unsqueeze(1)
        final_mask = diff_label_mask & same_batch_mask

        if not final_mask.any():
            return torch.tensor(
                0.0,
                device=label_embeddings.device,
                dtype=label_embeddings.dtype,
                requires_grad=True,
            )

        penalties = F.relu(sim_matrix[final_mask] - self.threshold)
        return penalties.mean()


class BCELoss(nn.Module):
    """Binary cross-entropy loss."""

    def __init__(self, config: GliZNetConfig):
        super().__init__()

    def forward(
        self, dense_logits: torch.Tensor, labels: torch.Tensor, **_
    ) -> torch.Tensor:
        mask = labels != -100
        if not mask.any():
            return torch.tensor(
                0.0,
                device=dense_logits.device,
                dtype=dense_logits.dtype,
                requires_grad=True,
            )

        valid_logits = dense_logits[mask]
        valid_targets = labels[mask]
        finite_mask = torch.isfinite(valid_logits)
        if not finite_mask.any():
            return torch.tensor(
                0.0,
                device=dense_logits.device,
                dtype=dense_logits.dtype,
                requires_grad=True,
            )

        return F.binary_cross_entropy_with_logits(
            valid_logits[finite_mask], valid_targets[finite_mask], reduction="mean"
        )


LOSS_REGISTRY: Dict[str, type] = {
    "softmax": SoftmaxLoss,
    "repulsion": RepulsionLoss,
    "bce": BCELoss,
}


class GliZNetLoss(nn.Module):
    """Orchestrates a configurable set of loss modules.

    Use ``GliZNetLoss.from_config(config)`` to build from a ``GliZNetConfig``.
    Individual loss modules receive all forward kwargs and ignore what they don't need.
    The returned dict contains each loss component (unweighted) plus ``"total"``
    (weighted sum) which callers should use for backpropagation.
    """

    def __init__(
        self,
        losses: nn.ModuleDict,
        weights: Dict[str, float],
        max_labels: int = 20,
    ):
        super().__init__()
        self.losses = losses
        self.weights = weights
        self.max_labels = max_labels

    @classmethod
    def from_config(cls, config: GliZNetConfig) -> "GliZNetLoss":
        weight_map = {
            "softmax": config.supcon_loss_weight,
            "repulsion": config.label_repulsion_weight,
            "bce": config.bce_loss_weight,
        }
        modules: Dict[str, nn.Module] = {}
        for name in config.losses:
            if name not in LOSS_REGISTRY:
                raise ValueError(
                    f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY)}"
                )
            modules[name] = LOSS_REGISTRY[name](config)
        weights = {name: weight_map.get(name, 1.0) for name in modules}
        return cls(nn.ModuleDict(modules), weights, max_labels=config.max_labels)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        batch_indices: torch.Tensor,
        label_ids: torch.Tensor,
        label_embeddings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        def _zero() -> torch.Tensor:
            return torch.tensor(
                0.0, device=logits.device, dtype=logits.dtype, requires_grad=True
            )

        result: Dict[str, torch.Tensor] = {name: _zero() for name in self.losses}

        if logits.numel() == 0:
            result["total"] = _zero()
            return result

        batch_size = labels.size(0)

        # Build dense logits matrix (B, max_labels) — fixed shape for torch.compile
        dense_logits = torch.full(
            (batch_size, self.max_labels),
            float("-inf"),
            device=logits.device,
            dtype=logits.dtype,
        )
        dense_logits[batch_indices, label_ids - 1] = logits.squeeze(-1)

        # Align labels to dense_logits shape
        valid_cols = min(labels.shape[1], self.max_labels)
        current_labels = labels[:, :valid_cols].to(logits.dtype)
        if current_labels.shape[1] < self.max_labels:
            padding = torch.full(
                (batch_size, self.max_labels - current_labels.shape[1]),
                -100.0,
                device=logits.device,
                dtype=logits.dtype,
            )
            current_labels = torch.cat([current_labels, padding], dim=1)

        context = {
            "dense_logits": dense_logits,
            "labels": current_labels,
            "label_embeddings": label_embeddings,
            "label_ids": label_ids,
            "batch_indices": batch_indices,
        }

        for name, module in self.losses.items():
            computed = module(**context)
            if torch.isnan(computed) or torch.isinf(computed):
                logger.warning(
                    f"NaN/Inf in {name} loss; zeroing for training stability."
                )
            else:
                result[name] = computed

        total = _zero()
        for name, loss_val in result.items():
            total = total + loss_val * self.weights.get(name, 1.0)
        result["total"] = total

        return result
