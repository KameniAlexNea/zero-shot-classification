from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_MAPPING


class GliZNetConfig(PretrainedConfig):
    """
    Configuration for `GliZNetForSequenceClassification` to support any HuggingFace transformer model.
    """

    pretrained_model_name_or_path: str
    projected_dim: int = None
    dropout_rate: float = 0.1
    similarity_metric: str = "dot"  # 'dot' or 'bilinear'
    temperature: float = 1.0
    model_type = "gliznet"

    def __init__(
        self,
        pretrained_model_name_or_path=None,
        projected_dim=None,
        dropout_rate=0.1,
        similarity_metric="dot",
        temperature=1.0,
        num_labels=1,
        **kwargs
    ):
        super().__init__(num_labels=num_labels, **kwargs)
        # Underlying transformer model identifier
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        # Optional projection dimension
        self.projected_dim = projected_dim
        # Dropout for transformer outputs
        self.dropout_rate = dropout_rate
        # Similarity metric: 'dot' or 'bilinear'
        self.similarity_metric = similarity_metric
        # Temperature scaling for logits
        self.temperature = temperature


class GliZNetPreTrainedModel(PreTrainedModel):
    config_class = GliZNetConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_sdpa = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif hasattr(module, "bias") and module.bias is not None:
            module.bias.data.zero_()


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


# Step 1: Register your config class with a unique model type name
CONFIG_MAPPING.register("gliznet", GliZNetConfig)

# Step 2: Register your model class
MODEL_MAPPING.register(GliZNetConfig, GliZNetPreTrainedModel)
