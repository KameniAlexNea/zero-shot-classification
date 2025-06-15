from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers import AutoModel


class GliZNetModel(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        hidden_size: int = None,
        dropout_rate: float = 0.1,
        similarity_metric: str = "bilinear",
        temperature: float = 1.0,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.config = self.encoder.config

        self.hidden_size = hidden_size or self.config.hidden_size
        self.similarity_metric = similarity_metric
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout_rate)

        self.proj = (
            nn.Linear(self.config.hidden_size, self.hidden_size)
            if self.hidden_size != self.config.hidden_size
            else nn.Identity()
        )

        if similarity_metric == "bilinear":
            self.bilinear = nn.Bilinear(self.hidden_size, self.hidden_size, 1)

        self.loss_fn = (
            nn.BCEWithLogitsLoss if self.similarity_metric != "cosine" else nn.MSELoss
        )()

    def compute_similarity(self, text_repr: torch.Tensor, label_repr: torch.Tensor):
        if self.similarity_metric == "cosine":
            sim = (
                F.cosine_similarity(text_repr, label_repr) + 1.0
            ) / 2  # Shift to [0, 1] range
            eps = 1e-7
            sim = sim.clamp(eps, 1 - eps)
        elif self.similarity_metric == "dot":
            sim = torch.mm(text_repr, label_repr.T)
        elif self.similarity_metric == "bilinear":
            sim = self.bilinear(text_repr, label_repr)
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")
        return sim / self.temperature

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label_mask: torch.Tensor,
        labels: list[torch.Tensor] = None,
    ):
        device = input_ids.device
        batch_size = input_ids.size(0)

        encoder_outputs = self.encoder(
            input_ids, attention_mask=attention_mask, return_dict=True
        )
        hidden = self.dropout(encoder_outputs.last_hidden_state)
        hidden_proj = self.proj(hidden)  # (batch_size, seq_len, hidden_size)

        # Get positions of label tokens
        pos = torch.nonzero(label_mask)[:, 0]  # (total_valid_samples,)
        labels_emb = hidden_proj[label_mask]  # (total_label_tokens, hidden_size)
        text_emb = hidden_proj[pos, 0]  # (total_label_tokens, hidden_size)

        # Compute similarities
        logits = self.compute_similarity(text_emb, labels_emb)

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
                sample_logits.reshape(1, -1)  # Reshape to (1, num_labels)
            )

            if labels is not None and sample_logits.numel() > 0:
                sample_labels = labels[i]
                if sample_labels.numel() == sample_logits.numel():
                    all_logits.append(sample_logits)
                    all_targets.append(sample_labels.view(-1, 1))

        output = {
            "loss": None,
            "logits": outputs_logits,
            "hidden_states": hidden_proj[:, 0],
        }

        if all_logits:
            total_logits = torch.cat(all_logits)
            total_labels = torch.cat(all_targets)
            output["loss"] = self.loss_fn(total_logits, total_labels)
        elif self.training:
            logger.warning(
                "No valid labels found in batch while training. Loss will not be computed."
            )

        return output

    def predict(
        self,
        input_ids,
        attention_mask,
        label_mask,
    ) -> list[list[float]]:
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, label_mask)
            results = []

            for logits in outputs["logits"]:
                probs = (
                    torch.sigmoid(logits)
                    if self.similarity_metric != "cosine"
                    else logits
                )
                results.append(probs.cpu().view(-1).numpy().tolist())
        return results
