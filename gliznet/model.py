from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModel, BertConfig, BertPreTrainedModel
from transformers.modeling_outputs import ModelOutput


class GliZNetConfig(BertConfig):
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
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[Union[List[torch.FloatTensor], torch.Tensor]] = None
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
            resized_embeddings: int = None,
        ):
            super().__init__(config)
            if similarity_metric not in ["dot", "bilinear", "dot_learning"]:
                raise ValueError(
                    f"Unsupported similarity metric: {similarity_metric}. Supported: 'dot', 'bilinear', 'dot_learning'."
                )
            setattr(self, self.base_model_prefix, AutoModel.from_config(config=config))
            self._initialize_config(
                config,
                projected_dim,
                similarity_metric,
                temperature,
                dropout_rate,
                resized_embeddings,
            )

            self.scale_loss = scale_loss
            self.dropout = nn.Dropout(self.config.dropout_rate)
            self._setup_layers()
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")
            self.post_init()

        def _initialize_config(
            self,
            config,
            projected_dim,
            similarity_metric,
            temperature,
            dropout_rate,
            resized_embeddings,
        ):
            config.projected_dim = getattr(config, "projected_dim", projected_dim)
            config.similarity_metric = getattr(
                config, "similarity_metric", similarity_metric
            )
            config.temperature = getattr(config, "temperature", temperature)
            config.dropout_rate = getattr(config, "dropout_rate", dropout_rate)
            config.resized_embeddings = getattr(
                config, "resized_embeddings", resized_embeddings
            )
            self.config = config

        def _setup_layers(self):
            projected_dim = self.config.projected_dim or self.config.hidden_size
            self.proj = (
                nn.Linear(self.config.hidden_size, projected_dim)
                if projected_dim != self.config.hidden_size
                else nn.Identity()
            )

            if self.config.similarity_metric == "bilinear":
                self.classifier = nn.Bilinear(projected_dim, projected_dim, 1)
            elif self.config.similarity_metric == "dot_learning":
                self.classifier = nn.Linear(projected_dim, 1)

            if self.config.resized_embeddings is not None and self.config.resized_embeddings != self.config.vocab_size:
                self.resize_token_embeddings(self.config.resized_embeddings)

        def resize_token_embeddings(self, new_num_tokens: int):
            print(new_num_tokens)
            base_model = getattr(self, self.base_model_prefix)
            if not hasattr(base_model, "resize_token_embeddings"):
                raise AttributeError(
                    f"{self.base_model_prefix} does not have a resize_token_embeddings method."
                )
            if hasattr(base_model, "resize_token_embeddings"):
                base_model.resize_token_embeddings(new_num_tokens)

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
            return_dict = (
                return_dict if return_dict is not None else self.config.use_return_dict
            )

            if lmask is None:
                raise ValueError(
                    "lmask is required for GliZNetForSequenceClassification"
                )

            hidden_states, attentions = self._get_hidden_states(
                input_ids,
                attention_mask,
                token_type_ids,
                output_attentions,
                output_hidden_states,
            )
            outputs_logits = self._compute_batch_logits(
                hidden_states, lmask, attentions
            )
            loss = (
                self._compute_loss(outputs_logits, labels)
                if labels is not None
                else None
            )

            if not return_dict:
                output = (outputs_logits, None)
                return ((loss,) + output) if loss is not None else output

            return GliZNetOutput(loss=loss, logits=outputs_logits, hidden_states=None)

        def compute_similarity(
            self, text_repr: torch.Tensor, label_repr: torch.Tensor
        ) -> torch.Tensor:
            if self.config.similarity_metric == "dot":
                return (text_repr * label_repr).mean(dim=1, keepdim=True)
            elif self.config.similarity_metric == "bilinear":
                return self.classifier(text_repr, label_repr)
            return self.classifier(text_repr * label_repr)

        def _get_hidden_states(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            output_attentions,
            output_hidden_states,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            # Only compute attention weights if explicitly requested
            compute_attention = output_attentions is not None and output_attentions
            encoder_outputs = self.backbone_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=compute_attention,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            cls_attn_weights = None
            if compute_attention and encoder_outputs.attentions is not None:
                cls_attn_weights = encoder_outputs.attentions[-1].mean(dim=1)[:, 0, :]
            return self.dropout(encoder_outputs.last_hidden_state), cls_attn_weights

        def _compute_batch_logits(
            self,
            hidden_states: torch.Tensor,  # (B, L, H)
            lmask: torch.Tensor,  # (B, L), values: 0 = text, 1... = label groups
            cls_attn_weights: Optional[torch.Tensor],  # (B, L)
        ) -> torch.Tensor:
            """
            Compute logits between [CLS] and averaged label embeddings per sample.
            """

            def compute_weighted(
                hidden_proj_raw: torch.Tensor,
                lmask_raw: torch.Tensor,
                cls_attn_weights_raw: torch.Tensor,
                label_id,
            ):
                attention_score = cls_attn_weights_raw[lmask_raw == label_id]
                return (
                    hidden_proj_raw[lmask_raw == label_id]
                    * attention_score.unsqueeze(1)
                ).sum(dim=0) / (attention_score.sum() + 1e-8)

            def compute_average(
                hidden_proj_raw: torch.Tensor,
                lmask_raw: torch.Tensor,
                label_id,
            ):
                label_tokens = hidden_proj_raw[lmask_raw == label_id]
                return label_tokens.mean(dim=0)

            hidden_proj: torch.Tensor = self.proj(hidden_states)  # (B, L, D)

            if cls_attn_weights is not None:
                # Use attention-weighted averaging
                all_logits = [
                    self.compute_similarity(
                        hidden_proj_raw[0],
                        torch.stack(
                            [
                                compute_weighted(
                                    hidden_proj_raw,
                                    lmask_raw,
                                    cls_attn_weights_raw,
                                    label_id,
                                )
                                for label_id in range(1, lmask_raw.max().item() + 1)
                            ]
                        ),
                    )
                    for hidden_proj_raw, lmask_raw, cls_attn_weights_raw in zip(
                        hidden_proj, lmask, cls_attn_weights
                    )
                ]
            else:
                # Use simple averaging
                all_logits = [
                    self.compute_similarity(
                        hidden_proj_raw[0],
                        torch.stack(
                            [
                                compute_average(
                                    hidden_proj_raw,
                                    lmask_raw,
                                    label_id,
                                )
                                for label_id in range(1, lmask_raw.max().item() + 1)
                            ]
                        ),
                    )
                    for hidden_proj_raw, lmask_raw in zip(hidden_proj, lmask)
                ]

            return torch.cat(all_logits)

        def _compute_loss(
            self, outputs_logits: torch.Tensor, labels: List[torch.Tensor]
        ) -> Optional[torch.Tensor]:
            total_labels = torch.cat([lab.view(-1, 1) for lab in labels])
            loss_values: torch.Tensor = self.loss_fn(outputs_logits, total_labels)
            return loss_values.mean() * self.scale_loss

        @torch.inference_mode()
        def predict(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            lmask: torch.Tensor,
            activation_fn: Optional[str] = "sigmoid",
        ) -> List[List[float]]:
            self.eval()
            outputs = self.forward(
                input_ids=input_ids, attention_mask=attention_mask, lmask=lmask
            )
            logits_list = (
                outputs.logits if isinstance(outputs, GliZNetOutput) else outputs[0]
            )
            logits_list = torch.split(logits_list, lmask.max(dim=1)[0].tolist())

            activate = (
                torch.sigmoid
                if activation_fn == "sigmoid"
                else lambda x: torch.softmax(x, dim=0)
            )
            return [
                activate(logits.squeeze(-1)).cpu().tolist() for logits in logits_list
            ]

    return GliZNetForSequenceClassification


GliZNetForSequenceClassification = create_gli_znet_for_sequence_classification()
