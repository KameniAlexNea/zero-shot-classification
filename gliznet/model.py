from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModel, BertConfig, BertPreTrainedModel
from transformers.modeling_outputs import ModelOutput


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
            config: BertConfig,
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
            config: BertConfig,
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
            """Compute logits for all samples in the batch efficiently."""
            device = hidden_states.device
            batch_size = hidden_states.size(0)

            # Get positions of label tokens more efficiently
            batch_indices, token_indices = torch.where(lmask)

            if len(batch_indices) == 0:
                return [torch.zeros((1, 0), device=device) for _ in range(batch_size)]

            # Project representations
            text_repr = self.proj(hidden_states[batch_indices, 0])  # CLS tokens
            label_repr = self.proj(
                hidden_states[batch_indices, token_indices]
            )  # Label tokens

            # Compute similarities
            logits = self.compute_similarity(text_repr, label_repr)

            # Group logits by batch index efficiently
            return self._group_logits_by_batch(logits, batch_indices, batch_size)

        def _group_logits_by_batch(
            self,
            logits: torch.Tensor,
            batch_indices: torch.Tensor,
            batch_size: int,
        ) -> List[torch.Tensor]:
            """Group logits by batch index using efficient tensor operations."""
            # Count occurrences per batch and split logits accordingly
            counts = torch.bincount(batch_indices, minlength=batch_size)
            # Split logits into chunks for each sample; zero-count yields empty tensor
            splits = torch.split(logits, counts.tolist(), dim=0)
            # Reshape each chunk to (1, num_labels) preserving device
            return [chunk.reshape(1, -1) for chunk in splits]

        def _compute_loss(
            self, outputs_logits: List[torch.Tensor], labels: List[torch.Tensor]
        ) -> Optional[torch.Tensor]:
            """Compute loss efficiently by batching valid samples."""
            # Flatten valid logits and labels in one pass
            filtered = [
                (log.view(-1, 1), lab.view(-1, 1))
                for log, lab in zip(outputs_logits, labels)
                if log.numel() > 0 and lab.numel() == log.numel()
            ]
            # Concatenate all valid logits and labels
            total_logits, total_labels = zip(*filtered)
            total_logits = torch.cat(total_logits)
            total_labels = torch.cat(total_labels)
            loss_values: torch.Tensor = self.loss_fn(total_logits, total_labels)
            return loss_values.mean() * self.scale_loss

        @torch.inference_mode()
        def predict(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            lmask: torch.Tensor,
            activation_fn: Optional[
                str
            ] = "sigmoid",  # Activation function for logits: sigmoid or softmax
        ) -> List[List[float]]:
            """
            Efficient prediction method for inference.

            Args:
                input_ids: Input token IDs
                attention_mask: Attention mask
                lmask: Label mask for identifying label positions

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

            # Vectorize activation: sigmoid via flat concat, softmax via masked padded tensor
            batch_size = len(logits_list)
            lengths = [log.numel() for log in logits_list]
            if batch_size == 0:
                return []
            # Sigmoid: independent per logit
            if activation_fn == "sigmoid":
                flat = (
                    torch.cat(logits_list)
                    if sum(lengths) > 0
                    else torch.tensor([], device=logits_list[0].device)
                )
                sig_flat = flat.sigmoid()
                splits = torch.split(sig_flat, lengths)
                return [s.cpu().tolist() for s in splits]
            # Softmax: mask padding as -inf to avoid affecting probabilities
            max_len = max(lengths) if lengths else 0
            if max_len == 0:
                return [[] for _ in range(batch_size)]
            padded = logits_list[0].new_full((batch_size, max_len), float("-inf"))
            for idx, log in enumerate(logits_list):
                if lengths[idx] > 0:
                    padded[idx, : lengths[idx]] = log
            probs_tensor = torch.softmax(padded, dim=-1)
            return [
                probs_tensor[i, :length].cpu().tolist()
                for i, length in enumerate(lengths)
            ]

    return GliZNetForSequenceClassification


GliZNetForSequenceClassification = create_gli_znet_for_sequence_classification()
