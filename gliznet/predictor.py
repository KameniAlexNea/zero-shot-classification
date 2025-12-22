from dataclasses import dataclass
from typing import List, Literal

import torch

from gliznet.model import GliZNetForSequenceClassification, GliZNetOutput
from gliznet.tokenizer import GliZNETTokenizer


@dataclass
class LabelScore:
    label: str
    score: float


@dataclass
class PredictionOutput:
    text: str
    labels: list[LabelScore]


class GliZNetPredictor:
    """Wrapper for zero-shot classification prediction."""

    def __init__(
        self,
        model_name_or_path: str = None,
        model: GliZNetForSequenceClassification = None,
        tokenizer: GliZNETTokenizer = None,
    ):
        """Initialize predictor with model and tokenizer.

        Args:
            model_name_or_path: HuggingFace model identifier or path
        """
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            if model_name_or_path is None:
                raise ValueError(
                    "Either model and tokenizer or model_name_or_path must be provided."
                )
            self.tokenizer = GliZNETTokenizer.from_pretrained(model_name_or_path)
            self.model = GliZNetForSequenceClassification.from_pretrained(
                model_name_or_path
            )
            self.model.eval()

    @torch.inference_mode()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        lmask: torch.Tensor,
        activation: Literal["sigmoid", "softmax"] = "sigmoid",
    ) -> List[List[float]]:
        """Predict label scores for each sample.

        Args:
            input_ids: Input token IDs (B, L)
            attention_mask: Attention mask (B, L)
            lmask: Label mask (B, L)
            activation: Activation function to apply

        Returns:
            List of score lists, one per sample
        """
        self.model.eval()

        outputs: GliZNetOutput = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            lmask=lmask,
            return_stats=True,
            return_dict=True,
        )

        logits = outputs.logits.squeeze(-1)
        batch_indices = outputs.batch_indices
        label_ids = outputs.label_ids

        # Apply activation
        if activation == "sigmoid":
            scores = torch.sigmoid(logits)
        else:
            scores = logits

        # Group by batch
        batch_size = input_ids.shape[0]
        results = [[] for _ in range(batch_size)]

        scores_list = scores.cpu().tolist()
        batch_list = batch_indices.cpu().tolist()
        label_list = label_ids.cpu().tolist()

        for batch_id in range(batch_size):
            # Collect scores for this batch
            batch_scores = [
                (label_id, score)
                for b, label_id, score in zip(batch_list, label_list, scores_list)
                if b == batch_id
            ]

            if batch_scores:
                # Sort by label_id
                batch_scores.sort(key=lambda x: x[0])
                final_scores = [score for _, score in batch_scores]

                # Apply softmax if requested
                if activation == "softmax":
                    final_scores = torch.softmax(
                        torch.tensor(final_scores), dim=0
                    ).tolist()

                results[batch_id] = final_scores

        return results

    @torch.inference_mode()
    def predict_example(
        self,
        text: str,
        labels: List[str],
        tokenizer: "GliZNETTokenizer",
        device: str = None,
        activation: Literal["sigmoid", "softmax"] = "sigmoid",
    ) -> PredictionOutput:
        """Predict scores for a single text with given labels.

        Args:
            text: Input text to classify
            labels: List of candidate labels
            tokenizer: GliZNETTokenizer instance
            device: Device to run inference on ("cpu" or "cuda")
            activation: Activation function to apply

        Returns:
            PredictionOutput instance mapping labels to scores
        """
        self.model.eval()
        if device is None:
            device = self.model.device

        # Tokenize
        batch = tokenizer([(text, labels)])

        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        lmask = batch["lmask"].to(device)

        # Get predictions
        predictions = self.predict(
            input_ids, attention_mask, lmask, activation=activation
        )
        scores = predictions[0]

        return PredictionOutput(
            text=text,
            labels=[
                LabelScore(label=label, score=score)
                for label, score in zip(labels, scores)
            ],
        )

    @torch.inference_mode()
    def predict_batch(
        self,
        texts: List[str],
        all_labels: List[List[str]],
        tokenizer: "GliZNETTokenizer",
        device: str = None,
        activation: Literal["sigmoid", "softmax"] = "sigmoid",
    ) -> List[PredictionOutput]:
        """Predict scores for multiple texts with their respective labels.

        Args:
            texts: List of input texts to classify
            all_labels: List of label lists (one per text)
            tokenizer: GliZNETTokenizer instance
            device: Device to run inference on ("cpu" or "cuda")
            activation: Activation function to apply
            return_sorted: If True, return results sorted by score (descending)

        Returns:
            List of PredictionOutput instances mapping labels to scores
        """
        self.model.eval()
        if device is None:
            device = self.model.device

        # Tokenize batch
        batch: dict[str, torch.Tensor] = tokenizer(
            [(text, labels) for text, labels in zip(texts, all_labels)]
        )

        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        lmask = batch["lmask"].to(device)

        # Get predictions
        predictions = self.predict(
            input_ids, attention_mask, lmask, activation=activation
        )

        # Build results
        results = [
            PredictionOutput(
                text=text,
                labels=[
                    LabelScore(label=label, score=score)
                    for label, score in zip(labels, scores)
                ],
            )
            for text, labels, scores in zip(texts, all_labels, predictions)
        ]

        return results
