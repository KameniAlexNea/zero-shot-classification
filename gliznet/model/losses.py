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
    """Scenario-adaptive focal loss with class-balanced per-sample averaging.

    Two improvements over standard focal loss for scenario-aware training:

    1. **Class-balanced averaging**: Within each sample, positive and negative
       losses are averaged separately, then combined with equal weight. This
       prevents minority-class dilution — in a needle sample (1 pos, 10 neg),
       the single positive gets 50% of the loss weight instead of 9%.

    2. **Adaptive gamma**: For pure-class samples (all-positive or all-negative),
       gamma is set to 0 (standard BCE). In these samples SoftmaxLoss returns 0
       because it needs both classes for contrastive learning. FocalLoss becomes
       the sole classification signal, so focal down-weighting is disabled to
       ensure reliable gradients. Mixed-class samples keep full gamma for
       hard-example mining.
    """

    def __init__(self, config: GliZNetConfig):
        super().__init__()
        self.gamma = config.focal_gamma

    def forward(
        self, dense_logits: torch.Tensor, labels: torch.Tensor, **_
    ) -> torch.Tensor:
        mask = labels != -100  # (B, K)
        if not mask.any():
            return dense_logits.new_zeros(1, requires_grad=True).squeeze()

        targets = torch.where(mask, labels, torch.zeros_like(labels))

        # p_t via sigmoid is numerically stable; torch.exp(-bce) underflows when
        # BCE is large (confident wrong prediction), zeroing the focal weight.
        probs = torch.sigmoid(dense_logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        bce = F.binary_cross_entropy_with_logits(
            dense_logits, targets, reduction="none"
        )  # (B, K)

        # Per-sample class detection
        pos_mask = (targets > 0.5) & mask  # (B, K)
        neg_mask = (targets < 0.5) & mask  # (B, K)
        n_pos = pos_mask.sum(dim=1)  # (B,)
        n_neg = neg_mask.sum(dim=1)  # (B,)
        has_both = (n_pos > 0) & (n_neg > 0)  # (B,)

        # Adaptive gamma: full focal for mixed samples, γ=0 (standard BCE) for
        # pure-class samples where SoftmaxLoss provides no signal.
        gamma = torch.where(has_both, self.gamma, 0.0).unsqueeze(1)  # (B, 1)
        focal_weight = (1.0 - p_t) ** gamma
        weighted_bce = focal_weight * bce * mask.float()  # (B, K)

        # Class-balanced per-sample loss: mean(pos_loss) and mean(neg_loss) get
        # equal weight regardless of class imbalance within the sample.
        pos_loss = (weighted_bce * pos_mask.float()).sum(dim=1) / n_pos.float().clamp(
            min=1
        )
        neg_loss = (weighted_bce * neg_mask.float()).sum(dim=1) / n_neg.float().clamp(
            min=1
        )

        n_classes = (n_pos > 0).float() + (n_neg > 0).float()
        sample_loss = (pos_loss + neg_loss) / n_classes.clamp(min=1)

        # Average over samples with at least one valid label
        valid_samples = mask.any(dim=1)
        if not valid_samples.any():
            return dense_logits.new_zeros(1, requires_grad=True).squeeze()

        return sample_loss[valid_samples].mean()


class AlignmentLoss(nn.Module):
    """Cosine alignment regularization between text and label embeddings.

    Forces cos(text_repr_i, label_i) to be high for positive pairs and low
    for negative pairs. This prevents the bilinear scorer from learning
    arbitrary projections that ignore embedding geometry.

    Uses CosineEmbeddingLoss (margin-based hinge): positives are pushed
    toward cos=1, negatives only need cos < margin to incur zero loss.
    """

    def __init__(self, config: GliZNetConfig):
        super().__init__()
        self.margin = 0.0

    def forward(
        self,
        label_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        batch_indices: torch.Tensor,
        label_ids: torch.Tensor,
        labels: torch.Tensor,
        **_,
    ) -> torch.Tensor:
        if label_embeddings.numel() == 0 or text_embeddings is None:
            return label_embeddings.new_zeros(1, requires_grad=True).squeeze()

        # Get ground truth for each (batch, label) pair
        targets = labels[batch_indices, label_ids - 1]
        valid = targets != -100
        if not valid.any():
            return label_embeddings.new_zeros(1, requires_grad=True).squeeze()

        targets = targets[valid].float()
        le = label_embeddings[valid]
        te = text_embeddings[valid]

        # Convert 0/1 targets to +1/-1 for CosineEmbeddingLoss convention
        y = 2 * targets - 1
        return F.cosine_embedding_loss(te, le, y, margin=self.margin)


LOSS_REGISTRY: Dict[str, type] = {
    "softmax": SoftmaxLoss,
    "repulsion": RepulsionLoss,
    "focal": FocalLoss,
    "alignment": AlignmentLoss,
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
            "alignment": config.alignment_loss_weight,
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
        text_embeddings: torch.Tensor = None,
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
            "text_embeddings": text_embeddings,
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
