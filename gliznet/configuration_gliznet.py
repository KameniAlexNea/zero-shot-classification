from transformers import PretrainedConfig

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
