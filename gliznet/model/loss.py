import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from gliznet.model.config import GliZNetConfig

logger = logging.getLogger(__name__)


class GliZNetLoss(nn.Module):
    """Combined loss: multi-label softmax, optional label repulsion, and decoupled BCE."""

    def __init__(self, config: GliZNetConfig):
        super().__init__()
        self.config = config
        # Learnable scale specifically for the auxiliary BCE loss (decoupled from main temperature)
        self.bce_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        batch_indices: torch.Tensor,
        label_ids: torch.Tensor,
        label_embeddings: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss.

        Args:
            logits: Predicted scores (N, 1) - already scaled by SimilarityHead
            labels: Ground truth labels (B, MaxLabels)
            batch_indices: Batch index for each logit (N,)
            label_ids: Label ID for each logit (N,)
            label_embeddings: Projected label embeddings (N, D)
            logit_scale: Current temperature scale from SimilarityHead

        Returns:
            Combined loss scalar
        """
        if logits.numel() == 0:
            return torch.tensor(
                0.0, device=logits.device, dtype=logits.dtype, requires_grad=True
            )

        batch_size = labels.size(0)
        max_label_id = self.config.max_labels

        if max_label_id == 0:
            return torch.tensor(
                0.0, device=logits.device, dtype=logits.dtype, requires_grad=True
            )

        # Reconstruct dense logits matrix (B, max_labels)
        dense_logits = torch.full(
            (batch_size, max_label_id),
            float("-inf"),
            device=logits.device,
            dtype=logits.dtype,
        )
        col_indices = label_ids - 1
        dense_logits[batch_indices, col_indices] = logits.squeeze(-1)

        # Align targets with dense_logits shape
        valid_cols = min(labels.shape[1], max_label_id)
        current_labels = labels[:, :valid_cols].to(logits.dtype)
        if current_labels.shape[1] < max_label_id:
            padding = torch.full(
                (batch_size, max_label_id - current_labels.shape[1]),
                -100.0,
                device=logits.device,
                dtype=logits.dtype,
            )
            current_labels = torch.cat([current_labels, padding], dim=1)

        total_loss = torch.tensor(
            0.0, device=logits.device, dtype=logits.dtype, requires_grad=True
        )

        # --- 1. Multi-label Softmax Loss (Primary) ---
        if self.config.supcon_loss_weight > 0:
            softmax_loss = self._multilabel_softmax_loss(dense_logits, current_labels)
            if torch.isnan(softmax_loss) or torch.isinf(softmax_loss):
                logger.warning(
                    "NaN/Inf in multilabel_softmax_loss; zeroing for training stability. "
                    "Check inputs and learning rate."
                )
                softmax_loss = torch.tensor(
                    0.0, device=logits.device, dtype=logits.dtype, requires_grad=True
                )
            total_loss = total_loss + softmax_loss * self.config.supcon_loss_weight

        # --- 2. Label Repulsion Loss (disabled by default) ---
        if self.config.label_repulsion_weight > 0:
            repulsion_loss = self._label_repulsion_loss(
                label_embeddings, label_ids, batch_indices
            )
            if torch.isnan(repulsion_loss) or torch.isinf(repulsion_loss):
                logger.warning(
                    "NaN/Inf in label_repulsion_loss; zeroing for training stability."
                )
                repulsion_loss = torch.tensor(
                    0.0, device=logits.device, dtype=logits.dtype, requires_grad=True
                )
            total_loss = (
                total_loss + repulsion_loss * self.config.label_repulsion_weight
            )

        # --- 3. Auxiliary BCE (Decoupled Temperature) ---
        if self.config.bce_loss_weight > 0:
            bce_loss = self._bce_loss(dense_logits, current_labels, logit_scale)
            if torch.isnan(bce_loss) or torch.isinf(bce_loss):
                logger.warning("NaN/Inf in bce_loss; zeroing for training stability.")
                bce_loss = torch.tensor(
                    0.0, device=logits.device, dtype=logits.dtype, requires_grad=True
                )
            total_loss = total_loss + bce_loss * self.config.bce_loss_weight

        return total_loss

    def _multilabel_softmax_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Multi-label softmax cross-entropy loss.

        For each sample, computes log-softmax over all label logits and maximises
        the mean log-probability of positive labels.  This is NOT Supervised
        Contrastive Loss (SupCon/InfoNCE): it operates on classification logits,
        not on embedding views, and has no contrastive pairs or anchor structure.
        """
        mask_valid = targets != -100
        targets_clean = targets.clone()
        targets_clean[~mask_valid] = 0.0

        has_positives = (targets_clean > 0.5).any(dim=1)
        if not has_positives.any():
            return torch.tensor(
                0.0, device=logits.device, dtype=logits.dtype, requires_grad=True
            )

        logits = logits[has_positives]
        targets_clean = targets_clean[has_positives]
        mask_valid_filtered = mask_valid[has_positives]

        has_valid_labels = mask_valid_filtered.any(dim=1)
        if not has_valid_labels.any():
            return torch.tensor(
                0.0, device=logits.device, dtype=logits.dtype, requires_grad=True
            )

        logits = logits[has_valid_labels]
        targets_clean = targets_clean[has_valid_labels]
        mask_valid_filtered = mask_valid_filtered[has_valid_labels]

        logits_masked = logits.clone()
        logits_masked[~mask_valid_filtered] = float("-inf")

        all_inf = torch.isinf(logits_masked).all(dim=1)
        if all_inf.any():
            valid_samples = ~all_inf
            if not valid_samples.any():
                return torch.tensor(
                    0.0, device=logits.device, dtype=logits.dtype, requires_grad=True
                )
            logits_masked = logits_masked[valid_samples]
            targets_clean = targets_clean[valid_samples]

        log_probs = F.log_softmax(logits_masked, dim=1)

        if torch.isnan(log_probs).any() or torch.isinf(log_probs).all():
            return torch.tensor(
                0.0, device=logits.device, dtype=logits.dtype, requires_grad=True
            )

        pos_mask = (targets_clean > 0.5).to(logits.dtype)
        # Use torch.where instead of multiplication to avoid -inf * 0 = NaN,
        # which occurs at padding positions where log_probs=-inf and pos_mask=0.
        sum_log_prob_pos = torch.where(
            pos_mask.bool(), log_probs, torch.zeros_like(log_probs)
        ).sum(dim=1)
        num_pos = pos_mask.sum(dim=1).clamp(min=1e-6)

        return (-sum_log_prob_pos / num_pos).mean()

    def _label_repulsion_loss(
        self,
        embeddings: torch.Tensor,
        label_ids: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize high cosine similarity between DIFFERENT labels in the same sample.

        Note: disabled by default (``label_repulsion_weight=0.0``).  Within-sample
        repulsion on contextual embeddings is conceptually unsound — semantically
        related labels conditioned on the same input *should* produce similar
        representations.  Only enable for static/non-contextual label embeddings.
        """
        if embeddings.numel() == 0:
            return torch.tensor(
                0.0,
                device=embeddings.device,
                dtype=embeddings.dtype,
                requires_grad=True,
            )

        embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
        sim_matrix = torch.matmul(embeddings_norm, embeddings_norm.T)

        diff_label_mask = label_ids.unsqueeze(0) != label_ids.unsqueeze(1)
        same_batch_mask = batch_indices.unsqueeze(0) == batch_indices.unsqueeze(1)
        final_mask = diff_label_mask & same_batch_mask

        if not final_mask.any():
            return torch.tensor(
                0.0,
                device=embeddings.device,
                dtype=embeddings.dtype,
                requires_grad=True,
            )

        penalties = F.relu(sim_matrix[final_mask] - self.config.repulsion_threshold)
        return penalties.mean()

    def _bce_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        main_logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Binary cross-entropy with decoupled temperature.

        Unscales the main logits and applies BCE-specific scaling.
        """
        mask = targets != -100
        if not mask.any():
            return torch.tensor(
                0.0, device=logits.device, dtype=logits.dtype, requires_grad=True
            )

        valid_logits = logits[mask]
        valid_targets = targets[mask]

        finite_mask = torch.isfinite(valid_logits)
        if not finite_mask.any():
            return torch.tensor(
                0.0, device=logits.device, dtype=logits.dtype, requires_grad=True
            )

        valid_logits = valid_logits[finite_mask]
        valid_targets = valid_targets[finite_mask]

        # Detach the main scale so BCE gradients do not flow back through logit_scale —
        # without detach() the "decoupled" temperature still receives gradient
        # contributions from the BCE path, making the decoupling illusory.
        scale_clamped = main_logit_scale.detach().clamp(-10, 10).exp().clamp(min=1e-6)
        raw_logits = valid_logits / scale_clamped
        bce_logits = raw_logits * self.bce_scale.abs().clamp(min=0.1, max=10.0)

        return F.binary_cross_entropy_with_logits(
            bce_logits, valid_targets, reduction="mean"
        )
