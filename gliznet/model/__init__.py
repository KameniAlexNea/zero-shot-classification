from transformers import AutoConfig, AutoModel

from gliznet.model.aggregator import CLSLabelAttentionAggregator, LabelAggregator
from gliznet.model.gliznet_modelling import (
    GliZNetForSequenceClassification,
    GliZNetPreTrainedModel,
)
from gliznet.model.config import GliZNetConfig
from gliznet.model.loss import GliZNetLoss
from gliznet.model.outputs import GliZNetOutput

# Register with HuggingFace Auto* classes so that
#   AutoConfig.from_pretrained(path)  →  GliZNetConfig
#   AutoModel.from_pretrained(path)   →  GliZNetForSequenceClassification
# and the "model type `gliznet`" warning disappears.
AutoConfig.register("gliznet", GliZNetConfig)
AutoModel.register(GliZNetConfig, GliZNetForSequenceClassification)

__all__ = [
    "GliZNetConfig",
    "GliZNetOutput",
    "CLSLabelAttentionAggregator",
    "LabelAggregator",
    "GliZNetLoss",
    "GliZNetPreTrainedModel",
    "GliZNetForSequenceClassification",
]
