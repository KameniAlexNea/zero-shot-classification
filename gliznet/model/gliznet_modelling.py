import logging
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel

from gliznet.model.aggregator import CLSLabelAttentionAggregator, LabelAggregator
from gliznet.model.config import GliZNetConfig
from gliznet.model.loss import GliZNetLoss
from gliznet.model.outputs import GliZNetOutput

if TYPE_CHECKING:
    from gliznet.tokenizer import GliZNETTokenizer

logger = logging.getLogger(__name__)


class GliZNetPreTrainedModel(PreTrainedModel):
    """Base class for GliZNet models."""

    backbone: PreTrainedModel
    config_class = GliZNetConfig
    base_model_prefix = "gliznet"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize weights using parent model's initialization."""
        if hasattr(self, "backbone") and hasattr(self.backbone, "_init_weights"):
            self.backbone._init_weights(module)


class GliZNetForSequenceClassification(GliZNetPreTrainedModel):
    """GliZNet model for zero-shot sequence classification.

    Architecture:
        - Backbone transformer (default: ModernBERT-base)
        - Token-level cross-attention to build a label-specific text representation
        - Bilinear head for scoring text-label pairs
    """

    def __init__(self, config: GliZNetConfig):
        super().__init__(config)
        self.config = config

        if config.backbone_config is None:
            # Lazy resolution: happens at model construction, not config instantiation,
            # so GliZNetConfig() remains a pure data object with no side effects.
            config.backbone_config = AutoConfig.from_pretrained(config.backbone_model)
        self.backbone: PreTrainedModel = AutoModel.from_config(config.backbone_config)

        aggregator_cls = (
            CLSLabelAttentionAggregator
            if config.use_cls_label_attention
            else LabelAggregator
        )
        self.aggregator = aggregator_cls(config)
        self.loss_fn = GliZNetLoss.from_config(config)

        self.post_init()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        """Resize token embeddings (for custom tokens)."""
        if new_num_tokens < self.config.backbone_config.vocab_size:
            return self.backbone
        if self.config.backbone_config.vocab_size != new_num_tokens:
            self.config.backbone_config.vocab_size = new_num_tokens
            self.backbone.config.vocab_size = new_num_tokens
        return self.backbone.resize_token_embeddings(new_num_tokens)

    @classmethod
    def from_backbone_pretrained(
        cls,
        config: GliZNetConfig,
        tokenizer: "GliZNETTokenizer",
        **kwargs,
    ) -> "GliZNetForSequenceClassification":
        """Create a new GliZNet model with pretrained backbone weights.

        Use this method when creating a NEW model (not loading a saved one).
        The backbone will be initialized with pretrained weights.

        Args:
            config: GliZNetConfig with backbone_model specified
            tokenizer: Tokenizer (used to resize embeddings for any added tokens)
            **kwargs: Additional arguments forwarded to AutoModel.from_pretrained

        Returns:
            GliZNet model with pretrained backbone
        """
        # Load backbone first so backbone_config is populated before model creation,
        # avoiding a redundant AutoConfig.from_pretrained() call inside __init__.
        pretrained_backbone: PreTrainedModel = AutoModel.from_pretrained(
            config.backbone_model, **kwargs
        )
        config.backbone_config = pretrained_backbone.config
        # Remove the backbone's dtype field — it reflects the original pretrained
        # model's serialization dtype, not the actual training dtype (e.g. bfloat16
        # from DeepSpeed).  Keeping it causes a dtype mismatch on reload.
        config.backbone_config.dtype = None
        model = cls(config)
        model.backbone.load_state_dict(pretrained_backbone.state_dict())
        model.resize_token_embeddings(len(tokenizer))
        return model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        lmask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_stats: bool = False,
        return_dict: bool = True,
        **_,
    ) -> Union[Tuple, GliZNetOutput]:
        """Forward pass.

        Args:
            input_ids: Input token IDs (B, L)
            attention_mask: Attention mask (B, L)
            lmask: Label mask, >0 for label tokens (B, L)
            labels: Ground truth labels (B, MaxLabels), -100 for padding
            return_dict: Whether to return a dict or tuple

        Returns:
            GliZNetOutput or tuple of (loss, logits, batch_indices, label_ids)
        """
        encoder_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_states = encoder_outputs.last_hidden_state

        (
            logits,
            batch_indices,
            label_ids,
            label_embeddings,
            text_embeddings,
        ) = self.aggregator(hidden_states, lmask, input_ids, attention_mask)

        loss = None
        output_logits = logits  # sparse (N_spans, 1) for inference

        if labels is not None:
            loss_dict = self.loss_fn(
                logits=logits,
                labels=labels,
                batch_indices=batch_indices,
                label_ids=label_ids,
                label_embeddings=label_embeddings,
            )
            loss = loss_dict["total"]
            # Reconstruct dense (B, max_labels) logits so the Trainer can all_gather
            # fixed-shape tensors in DDP eval. Unused positions filled with -100.0.
            if logits.numel() > 0:
                batch_size, max_labels = labels.shape
                output_logits = logits.new_full((batch_size, max_labels), -100.0)
                col_idx = label_ids - 1  # 1-indexed → 0-indexed
                in_range = col_idx < max_labels
                output_logits[batch_indices[in_range], col_idx[in_range]] = (
                    logits.squeeze(-1)[in_range]
                )

        if not return_dict:
            return (loss, output_logits, batch_indices, label_ids)

        return GliZNetOutput(
            loss=loss,
            logits=output_logits,
            batch_indices=batch_indices if return_stats else None,
            label_ids=label_ids if return_stats else None,
            label_embeddings=label_embeddings if return_stats else None,
            text_embeddings=text_embeddings if return_stats else None,
        )

    @torch.inference_mode()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        lmask: torch.Tensor,
    ) -> List[List[float]]:
        """Run inference and return sigmoid scores grouped by batch item.

        Args:
            input_ids: (B, L)
            attention_mask: (B, L)
            lmask: (B, L)

        Returns:
            List of length B; each element is a list of sigmoid scores for
            that sample's labels in ascending label-ID order.
        """
        out = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            lmask=lmask,
            return_stats=True,
        )
        batch_size = input_ids.shape[0]
        results: List[List[float]] = [[] for _ in range(batch_size)]

        if out.logits is None or out.logits.numel() == 0:
            return results

        scores = torch.sigmoid(out.logits.squeeze(-1))
        batch_indices = out.batch_indices
        label_ids = out.label_ids

        order = torch.argsort(
            batch_indices * (int(label_ids.max().item()) + 1) + label_ids
        )
        for idx in order.tolist():
            results[batch_indices[idx].item()].append(scores[idx].item())

        return results
