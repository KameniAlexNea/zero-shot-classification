from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import PretrainedConfig
from transformers.modeling_outputs import ModelOutput


class GliZNetConfig(PretrainedConfig):
    """
    Configuration for `GliZNetForSequenceClassification` to support any HuggingFace transformer model.
    """
    model_type = "gliznet"

    def __init__(
        self,
        base_model_name=None,
        projected_dim=None,
        dropout_rate=0.1,
        similarity_metric="dot",
        temperature=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.projected_dim = projected_dim
        self.dropout_rate = dropout_rate
        # Similarity metric: 'dot' or 'bilinear'
        self.similarity_metric = similarity_metric
        # Temperature scaling for logits
        self.temperature = temperature
        self.base_model_name = base_model_name


@dataclass
class GliZNetOutput(ModelOutput):
    """
    Output type of [`GliZNetForSequenceClassification`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification loss.
        logits (`list[torch.FloatTensor]`):
            Classification scores for each sample in the batch.
        hidden_states (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Hidden states of the model at the output of the encoder (CLS token representations).
    """

    loss: Optional[torch.FloatTensor] = None
    logits: List[torch.FloatTensor] = None
    hidden_states: Optional[torch.FloatTensor] = None
