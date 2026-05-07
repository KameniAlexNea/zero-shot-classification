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
        targets_clean = torch.where(mask_valid, labels, torch.zeros_like(labels))

        pos_mask = targets_clean > 0.5
        has_positives = pos_mask.any(dim=1)
        if not has_positives.any():
            return dense_logits.new_zeros(1, requires_grad=True).squeeze()

        logits = dense_logits[has_positives]
        pos_mask = pos_mask[has_positives]
        mask_valid = mask_valid[has_positives]
        neg_mask = mask_valid & ~pos_mask

        neg_logits = logits.masked_fill(~neg_mask, -1e9)
        if self.margin > 0.0:
            neg_logits = neg_logits + self.margin

        # Skip samples with no negatives — logsumexp over all -1e9 adds fake negative
        # mass and distorts the denominator even if numerically small.
        has_negatives = neg_mask.any(dim=1)
        if not has_negatives.any():
            return dense_logits.new_zeros(1, requires_grad=True).squeeze()
        logits = logits[has_negatives]
        pos_mask = pos_mask[has_negatives]
        neg_logits = neg_logits[has_negatives]

        neg_lse = torch.logsumexp(neg_logits, dim=1)

        neg_lse_exp = neg_lse.unsqueeze(1).expand_as(logits)
        denom_lse = torch.logsumexp(torch.stack([logits, neg_lse_exp], dim=2), dim=2)
        per_pos_loss = denom_lse - logits

        valid_pos = pos_mask
        per_pos_loss = per_pos_loss.masked_fill(~valid_pos, 0.0)

        num_pos = valid_pos.sum(dim=1).float().clamp(min=1e-6)
        return (per_pos_loss.sum(dim=1) / num_pos).mean()


class RepulsionLoss(nn.Module):
    """Per-sample VICReg-style regularization to prevent label embedding collapse.

    Computes variance and covariance terms WITHIN each sample's labels
    independently (not globally across the batch), preserving contextual
    sensitivity — the same label can have different embeddings in different
    text contexts.

    Fully vectorized via dense (B, K, D) tensor and batched matmul.

    Reference: Bardes et al., "VICReg: Variance-Invariance-Covariance
    Regularization for Self-Supervised Learning", ICLR 2022.
    """

    def __init__(self, config: GliZNetConfig):
        super().__init__()
        self.variance_target = 0.05
        self.eps = 1e-4
        self.covariance_weight = 0.04
        self.max_labels = config.max_labels

    def forward(
        self,
        label_embeddings: torch.Tensor,
        batch_indices: torch.Tensor,
        label_ids: torch.Tensor,
        **_,
    ) -> torch.Tensor:
        if label_embeddings.numel() == 0 or label_embeddings.shape[0] < 2:
            return label_embeddings.new_zeros(1, requires_grad=True).squeeze()

        D = label_embeddings.shape[1]
        device = label_embeddings.device
        B = batch_indices.max().item() + 1
        K = self.max_labels

        # L2-normalize (detached magnitude) so variance/covariance operate in
        # directional space only.
        mag = label_embeddings.norm(dim=-1, keepdim=True).detach().clamp(min=1e-6)
        normalized = label_embeddings / mag

        # Build dense (B, K, D) tensor — zeros for empty slots
        dense = normalized.new_zeros(B, K, D)
        dense[batch_indices, label_ids - 1] = normalized

        # Validity mask (B, K) and per-sample label counts
        valid = torch.zeros(B, K, dtype=torch.bool, device=device)
        valid[batch_indices, label_ids - 1] = True
        counts = valid.sum(dim=1).float()  # (B,)

        # Only compute for samples with >= 2 labels
        multi_label = counts >= 2
        if not multi_label.any():
            return label_embeddings.new_zeros(1, requires_grad=True).squeeze()

        dense = dense[multi_label]  # (B', K, D)
        valid_m = valid[multi_label]  # (B', K)
        counts_m = counts[multi_label]  # (B',)
        B_eff = dense.shape[0]

        # Float mask for arithmetic: (B', K, 1)
        fmask = valid_m.unsqueeze(-1).float()

        # Per-sample mean: (B', 1, D)
        mean = (dense * fmask).sum(dim=1, keepdim=True) / counts_m.view(-1, 1, 1)

        # Center (zeroing invalid positions)
        centered = (dense - mean) * fmask  # (B', K, D)

        # ── Variance term: per-sample, per-dimension std ──
        # Var = sum((x - mu)^2) / (N - 1) for each sample & dim
        var = (centered.pow(2)).sum(dim=1) / (counts_m.unsqueeze(1) - 1)  # (B', D)
        std = torch.sqrt(var + self.eps)  # (B', D)
        variance_loss = F.relu(self.variance_target - std).mean()

        # ── Covariance term: per-sample off-diagonal covariance ──
        # cov_b = centered_b^T @ centered_b / (N_b - 1)  →  (D, D) per sample
        # Using bmm: (B', D, K) @ (B', K, D) → (B', D, D)
        cov = torch.bmm(centered.transpose(1, 2), centered)  # (B', D, D)
        cov = cov / (counts_m.view(-1, 1, 1) - 1)

        # Zero diagonal (we only penalize off-diagonal correlations)
        diag_mask = torch.eye(D, dtype=torch.bool, device=device).unsqueeze(0)
        cov = cov.masked_fill(diag_mask, 0.0)

        # Mean of squared off-diagonal entries, averaged over samples
        covariance_loss = cov.pow(2).sum(dim=(1, 2)).mean() / D

        return variance_loss + self.covariance_weight * covariance_loss



class FocalLoss(nn.Module):
    """Focal loss — down-weights easy examples to focus on hard ones."""

    def __init__(self, config: GliZNetConfig):
        super().__init__()
        self.gamma = config.focal_gamma

    def forward(
        self, dense_logits: torch.Tensor, labels: torch.Tensor, **_
    ) -> torch.Tensor:
        mask = labels != -100
        if not mask.any():
            return dense_logits.new_zeros(1, requires_grad=True).squeeze()

        valid_logits = dense_logits[mask]
        valid_targets = labels[mask]

        # p_t via sigmoid is numerically stable; torch.exp(-bce) underflows when
        # BCE is large (confident wrong prediction), zeroing the focal weight.
        probs = torch.sigmoid(valid_logits)
        p_t = probs * valid_targets + (1 - probs) * (1 - valid_targets)
        focal_weight = (1.0 - p_t) ** self.gamma

        bce = F.binary_cross_entropy_with_logits(
            valid_logits, valid_targets, reduction="none"
        )
        return (focal_weight * bce).mean()


LOSS_REGISTRY: Dict[str, type] = {
    "softmax": SoftmaxLoss,
    "repulsion": RepulsionLoss,
    "focal": FocalLoss,
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
            "focal": config.focal_loss_weight,
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
        result: Dict[str, torch.Tensor] = {}

        if logits.numel() == 0:
            zero = logits.new_zeros(1, requires_grad=True).squeeze()
            for name in self.losses:
                result[name] = zero
            result["total"] = zero
            return result

        batch_size = labels.size(0)

        # Build dense logits matrix (B, max_labels) — fixed shape for torch.compile
        dense_logits = torch.full(
            (batch_size, self.max_labels),
            -1e9,
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

        total = logits.new_zeros(1).squeeze()
        for name, module in self.losses.items():
            computed = module(**context)
            # Use nan_to_num (no GPU sync) instead of isnan/isinf checks
            computed = torch.nan_to_num(computed, nan=0.0, posinf=0.0, neginf=0.0)
            result[name] = computed
            total = total + computed * self.weights.get(name, 1.0)

        result["total"] = total

        return result
