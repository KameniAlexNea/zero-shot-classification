from collections import defaultdict
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from loguru import logger
from transformers import AutoModel, PreTrainedModel, AutoConfig

from .config import GliZNetConfig, GliZNetOutput


class GliZNetForSequenceClassification(PreTrainedModel):
    # Associate this model with the custom config class
    config_class = GliZNetConfig

    def __init__(
        self,
        config: GliZNetConfig,
    ):
        super().__init__(config)
        # load any pretrained transformer model and alias for backward compatibility
        self.model = AutoModel.from_pretrained(config.base_model_name)
        self.config = config

        # Model parameters
        self.projected_dim = config.projected_dim
        self.similarity_metric = config.similarity_metric
        self.temperature = config.temperature
        self.dropout = nn.Dropout(config.dropout_rate)

        # Projection layer
        if (
            self.projected_dim != self.config.hidden_size
            and self.projected_dim is not None
        ):
            self.proj = nn.Linear(self.config.hidden_size, self.projected_dim)
        else:
            self.proj = nn.Identity()

        # Similarity computation layers
        if self.similarity_metric == "bilinear":
            self.classifier = nn.Bilinear(self.projected_dim, self.projected_dim, 1)

        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        # Initialize weights and apply final processing
        self.post_init()

    def compute_similarity(self, text_repr: torch.Tensor, label_repr: torch.Tensor):
        if self.similarity_metric == "dot":
            sim = (text_repr * label_repr).sum(dim=1, keepdim=True) / text_repr.shape[1]
        elif self.similarity_metric == "bilinear":
            sim = self.classifier(text_repr, label_repr)
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")
        return sim

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        label_mask: Optional[torch.Tensor] = None,
        labels: Optional[List[torch.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], GliZNetOutput]:
        r"""
        labels (`list[torch.Tensor]`, *optional*):
            Labels for computing the classification loss. Each tensor in the list corresponds to labels for one sample.
        label_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to identify label token positions.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if label_mask is None:
            raise ValueError(
                "label_mask is required for GliZNetForSequenceClassification"
            )

        device = input_ids.device
        batch_size = input_ids.size(0)

        # Get encoder outputs
        encoder_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden = self.dropout(encoder_outputs.last_hidden_state)
        hidden_proj = self.proj(hidden)  # (batch_size, seq_len, hidden_size)

        # Get positions of label tokens
        pos = torch.nonzero(label_mask)[:, 0]  # (total_valid_samples,)

        # Compute similarities: (total_label_tokens, hidden_size) x (total_label_tokens, hidden_size)
        logits = self.compute_similarity(hidden_proj[pos, 0], hidden_proj[label_mask])

        # Group logits by batch
        grouped_logits = defaultdict(list)
        for i, logit in zip(pos.tolist(), logits):
            grouped_logits[i].append(logit)

        outputs_logits = []
        all_logits = []
        all_targets = []

        for i in range(batch_size):
            if i in grouped_logits:
                sample_logits = torch.stack(grouped_logits[i])
            else:
                sample_logits = torch.zeros((0,), device=device)

            outputs_logits.append(
                sample_logits.reshape(1, -1)
            )

            if labels is not None and sample_logits.numel() > 0:
                sample_labels = labels[i]
                if sample_labels.numel() == sample_logits.numel():
                    all_logits.append(sample_logits.view(-1, 1))
                    all_targets.append(sample_labels.view(-1, 1))

        # Compute loss
        loss = None
        if all_logits:
            total_logits = torch.cat(all_logits)
            total_labels = torch.cat(all_targets)
            loss_values: torch.Tensor = self.loss_fn(total_logits, total_labels)
            loss = loss_values.mean(dim=0, keepdim=True)  # Average loss across all valid labels
        elif self.training:
            logger.warning(
                "No valid labels found in batch while training. Loss will not be computed."
            )

        # Get CLS token representations for hidden states
        cls_hidden_states = hidden_proj[:, 0]  # (batch_size, hidden_size)

        if not return_dict:
            output = (outputs_logits, cls_hidden_states)
            return ((loss,) + output) if loss is not None else output

        return GliZNetOutput(
            loss=loss,
            logits=outputs_logits,
            hidden_states=cls_hidden_states,
        )

    def predict(
        self,
        input_ids,
        attention_mask,
        label_mask,
    ) -> List[List[float]]:
        """
        Prediction method for inference.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                label_mask=label_mask,
            )
            results = []

            logits_list = (
                outputs.logits if isinstance(outputs, GliZNetOutput) else outputs[0]
            )
            for logits in logits_list:
                results.append(logits.sigmoid().cpu().view(-1).numpy().tolist())
        return results
