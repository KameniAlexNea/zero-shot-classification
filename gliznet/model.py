from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModel, BertConfig, BertPreTrainedModel
from transformers.modeling_outputs import ModelOutput


class GliZNetConfig(BertConfig):
    """
    Configuration class for GliZNet model.

    Args:
        projected_dim: Dimension of the projected representations (default: None)
        similarity_metric: Similarity metric to use ('dot' or 'bilinear', default: 'dot')
        temperature: Temperature scaling factor for similarity (default: 1.0)
        dropout_rate: Dropout rate for the model (default: 0.1)
    """

    def __init__(
        self,
        projected_dim: Optional[int] = None,
        similarity_metric: str = "dot",
        temperature: float = 1.0,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.projected_dim = projected_dim
        self.similarity_metric = similarity_metric
        self.temperature = temperature
        self.dropout_rate = dropout_rate


@dataclass
class GliZNetOutput(ModelOutput):
    """
    Output type of [`GliZNetForSequenceClassification`].

    Args:
        loss: Classification loss (shape: (1,)) when labels are provided
        logits: List of classification scores for each sample in the batch
        hidden_states: Hidden states of the model (currently unused/None)
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[torch.FloatTensor] = None


def create_gli_znet_for_sequence_classification(base_class=BertPreTrainedModel):
    class GliZNetForSequenceClassification(base_class):
        def __init__(
            self,
            config: base_class.config_class,
            projected_dim: Optional[int] = None,
            similarity_metric: str = "dot",
            temperature: float = 1.0,
            dropout_rate: float = 0.1,
            scale_loss: float = 10.0,
        ):
            super().__init__(config)
            # Load any pretrained transformer model and alias for backward compatibility
            setattr(self, self.base_model_prefix, AutoModel.from_config(config=config))

            # Initialize configuration parameters
            self._initialize_config(
                config, projected_dim, similarity_metric, temperature, dropout_rate
            )

            # Model parameters
            self.scale_loss = scale_loss
            self.dropout = nn.Dropout(self.config.dropout_rate)

            # Setup projection and similarity layers
            self._setup_layers()

            self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

            # Initialize weights and apply final processing
            self.post_init()

        def _initialize_config(
            self,
            config: GliZNetConfig,
            projected_dim: Optional[int],
            similarity_metric: str,
            temperature: float,
            dropout_rate: float,
        ) -> None:
            """Initialize configuration parameters with backward compatibility."""
            if not hasattr(config, "projected_dim"):  # New config
                config.projected_dim = projected_dim
                config.similarity_metric = similarity_metric
                config.temperature = temperature
                config.dropout_rate = dropout_rate
            # For backward compatibility, config already has these attributes
            self.config = config

        def _setup_layers(self) -> None:
            """Setup projection and similarity computation layers."""
            projected_dim = self.config.projected_dim or self.config.hidden_size

            # Projection layer
            if projected_dim != self.config.hidden_size:
                self.proj = nn.Linear(self.config.hidden_size, projected_dim)
            else:
                self.proj = nn.Identity()

            # Similarity computation layers
            if self.config.similarity_metric == "bilinear":
                self.classifier = nn.Bilinear(projected_dim, projected_dim, 1)

        def compute_similarity(
            self, text_repr: torch.Tensor, label_repr: torch.Tensor
        ) -> torch.Tensor:
            """Compute similarity between text and label representations."""
            if self.config.similarity_metric == "dot":
                return (text_repr * label_repr).mean(dim=1, keepdim=True)
            elif self.config.similarity_metric == "bilinear":
                return self.classifier(text_repr, label_repr)
            else:
                raise ValueError(
                    f"Unsupported similarity metric: {self.config.similarity_metric}"
                )

        def backbone_forward(self, *args, **kwargs):
            return getattr(self, self.base_model_prefix)(*args, **kwargs)

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            lmask: Optional[torch.Tensor] = None,
            labels: Optional[List[torch.Tensor]] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple[torch.Tensor], GliZNetOutput]:
            """
            Forward pass for GliZNet sequence classification.

            Args:
                lmask: Mask to identify label token positions (batch_size, sequence_length)
                labels: List of label tensors for computing classification loss
            """
            return_dict = (
                return_dict if return_dict is not None else self.config.use_return_dict
            )

            if lmask is None:
                raise ValueError(
                    "lmask is required for GliZNetForSequenceClassification"
                )

            # Get encoder outputs
            hidden_states = self._get_hidden_states(
                input_ids,
                attention_mask,
                token_type_ids,
                output_attentions,
                output_hidden_states,
            )

            # Compute logits for each sample
            outputs_logits = self._compute_batch_logits(hidden_states, lmask)

            # Compute loss if labels are provided
            loss = (
                self._compute_loss(outputs_logits, labels)
                if labels is not None
                else None
            )

            if not return_dict:
                output = (outputs_logits, None)  # cls_hidden_states removed
                return ((loss,) + output) if loss is not None else output

            return GliZNetOutput(loss=loss, logits=outputs_logits, hidden_states=None)

        def _get_hidden_states(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            token_type_ids: Optional[torch.Tensor],
            output_attentions: Optional[bool],
            output_hidden_states: Optional[bool],
        ) -> torch.Tensor:
            """Get hidden states from the backbone model."""
            encoder_outputs = self.backbone_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            return self.dropout(encoder_outputs.last_hidden_state)

        def _compute_batch_logits(
            self, hidden_states: torch.Tensor, lmask: torch.Tensor
        ) -> List[torch.Tensor]:
            """Compute logits for all samples in the batch with vectorized operations."""
            # Only project CLS and label tokens to save memory
            mask = lmask.bool()
            counts = mask.sum(dim=1)

            # Project CLS representations only
            cls_proj: torch.Tensor = self.proj(
                hidden_states[:, 0, :]
            )  # (batch_size, proj_dim)

            # Project label token representations only
            lab_proj: torch.Tensor = self.proj(
                hidden_states[mask]
            )  # (total_labels, proj_dim)

            # Compute similarity
            cls_flat = cls_proj.repeat_interleave(counts, dim=0)
            if self.config.similarity_metric == "dot":
                sims = (cls_flat * lab_proj).mean(dim=1)
            else:
                sims = self.classifier(cls_flat, lab_proj).view(-1)

            # Split sims per sample and return list of (1, num_labels) tensors
            splits = torch.split(sims, counts.tolist())
            return [i.view(-1, 1) for i in splits]

        def _compute_loss(
            self, outputs_logits: List[torch.Tensor], labels: List[torch.Tensor]
        ) -> Optional[torch.Tensor]:
            """Compute loss efficiently by batching valid samples."""
            total_logits = torch.cat(outputs_logits)
            total_labels = torch.cat([lab.view(-1, 1) for lab in labels])
            loss_values: torch.Tensor = self.loss_fn(total_logits, total_labels)
            return loss_values.mean() * self.scale_loss

        @torch.inference_mode()
        def predict(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            lmask: torch.Tensor,
            activation_fn: Optional[str] = "sigmoid",
        ) -> List[List[float]]:
            """
            Prediction method for inference.

            Args:
                input_ids: Input token IDs
                attention_mask: Attention mask
                lmask: Label mask for identifying label positions
                activation_fn: Activation function ('sigmoid' or 'softmax')

            Returns:
                List of probability scores for each sample
            """
            self.eval()
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                lmask=lmask,
            )

            logits_list = (
                outputs.logits if isinstance(outputs, GliZNetOutput) else outputs[0]
            )

            if not logits_list:
                return []

            # Apply activation function to each tensor in the list
            probs_list = []

            def activate(x):
                return (
                    torch.sigmoid(x)
                    if activation_fn == "sigmoid"
                    else torch.softmax(x, dim=0)
                )

            probs_list = [
                activate(logits.squeeze(-1))
                .cpu()
                .tolist()  # logits shape: (num_labels, 1) -> squeeze to (num_labels,)
                for logits in logits_list
            ]
            return probs_list

    return GliZNetForSequenceClassification


GliZNetForSequenceClassification = create_gli_znet_for_sequence_classification()
