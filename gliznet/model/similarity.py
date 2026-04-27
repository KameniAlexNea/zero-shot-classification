import torch
import torch.nn as nn
import torch.nn.functional as F

from gliznet.model.config import GliZNetConfig


class SimilarityHead(nn.Module):
    """Computes similarity between text and label representations."""

    def __init__(self, config: GliZNetConfig, projected_dim: int):
        super().__init__()
        self.config = config

        # Learnable temperature for scaling logits
        self.logit_scale = nn.Parameter(
            torch.tensor(config.logit_scale_init, dtype=torch.float32)
        )

        if config.similarity_metric == "bilinear":
            self.classifier = nn.Bilinear(projected_dim, projected_dim, 1)
        elif config.similarity_metric not in ("dot", "cosine"):
            raise ValueError(
                f"Unknown similarity_metric: {config.similarity_metric}. "
                "Choose 'dot', 'bilinear', or 'cosine'."
            )

    def forward(
        self, text_repr: torch.Tensor, label_repr: torch.Tensor
    ) -> torch.Tensor:
        """Compute similarity scores.

        Args:
            text_repr: Text representations (N, D)
            label_repr: Label representations (N, D)

        Returns:
            Scaled similarity scores (N, 1)
        """
        if self.config.similarity_metric == "bilinear":
            logits = self.classifier(text_repr, label_repr)
        elif self.config.similarity_metric == "dot":
            # True element-wise dot product, scaled by learnable temperature
            scale = self.logit_scale.clamp(-10, 10).exp()
            logits = (text_repr * label_repr).sum(dim=-1, keepdim=True) * scale
        else:  # cosine
            text_norm = F.normalize(text_repr, p=2, dim=-1)
            label_norm = F.normalize(label_repr, p=2, dim=-1)
            raw_sim = (text_norm * label_norm).sum(dim=-1, keepdim=True)
            scale = self.logit_scale.clamp(-10, 10).exp()
            logits = raw_sim * scale

        return logits
