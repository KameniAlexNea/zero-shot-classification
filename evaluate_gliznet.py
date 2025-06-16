import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from gliznet.model import GliZNetModel
from gliznet.tokenizer import GliZNETTokenizer, load_dataset


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""

    model_path: str = "results/checkpoint-3228"
    model_name: str = "results/checkpoint-3228"
    device: str = "auto"
    batch_size: int = 64
    max_labels: int = 20
    threshold: float = 0.5
    results_dir: str = "results/evaluation"


class ModelEvaluator:
    """Handles model evaluation with comprehensive metrics."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = self._get_device()
        self.model, self.tokenizer = self._load_models()

    def _get_device(self) -> torch.device:
        """Get the appropriate device for inference."""
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config.device)

    def _load_models(self) -> Tuple[GliZNetModel, GliZNETTokenizer]:
        """Load model and tokenizer from checkpoint."""
        logger.info(f"Loading model from {self.config.model_path} on {self.device}")

        try:
            tokenizer = GliZNETTokenizer.from_pretrained(self.config.model_name)

            model = model = GliZNetModel.from_pretrained(
                self.config.model_name,
                projected_dim=256,
                similarity_metric="dot",
            )
            model.to(self.device)
            model.eval()

            logger.info("Model loaded successfully")
            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _prepare_batch_inputs(
        self, sentences: List[str], labels: List[List[str]], masks: List[List[int]]
    ) -> Tuple[List[str], List[List[str]], List[List[int]]]:
        """Prepare batch inputs by chunking labels if needed."""
        inputs_texts = []
        labels_texts = []
        labels_logits = []

        for sentence, label, mask in zip(sentences, labels, masks):
            for i in range(0, len(label), self.config.max_labels):
                inputs_texts.append(sentence)
                labels_texts.append(label[i : i + self.config.max_labels])
                labels_logits.append(mask[i : i + self.config.max_labels])

        return inputs_texts, labels_texts, labels_logits

    def predict_batch(
        self, inputs_texts: List[str], labels_texts: List[List[str]]
    ) -> List[float]:
        """Make predictions for a batch of inputs."""
        inputs = self.tokenizer(
            inputs_texts,
            labels=labels_texts,
            pad=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            inputs_gpu = {k: v.to(self.device) for k, v in inputs.items()}
            predictions = self.model.predict(**inputs_gpu)

        return predictions

    def evaluate_dataset(self, dataset) -> Dict[str, Any]:
        """Evaluate the model on the given dataset."""
        logger.info("Starting evaluation...")

        all_predictions = []
        all_true_labels = []
        all_pred_scores = []
        detailed_results = []

        for batch in tqdm(
            dataset.iter(batch_size=self.config.batch_size), desc="Evaluating"
        ):
            try:
                sentences = batch["text"]
                labels = batch["labels_text"]
                masks = batch["labels_int"]

                inputs_texts, labels_texts, labels_logits = self._prepare_batch_inputs(
                    sentences, labels, masks
                )

                predictions = self.predict_batch(inputs_texts, labels_texts)

                # Store results for metrics calculation
                for sentence, label_list, mask_list, pred_list in zip(
                    inputs_texts, labels_texts, labels_logits, predictions
                ):
                    binary_preds = [
                        1 if p >= self.config.threshold else 0 for p in pred_list
                    ]

                    all_predictions.extend(binary_preds[: len(mask_list)])
                    all_true_labels.extend(mask_list)
                    all_pred_scores.extend(pred_list[: len(mask_list)])

                    detailed_results.append(
                        {
                            "sentence": sentence,
                            "labels": label_list,
                            "true_labels": mask_list,
                            "predicted_scores": pred_list[: len(mask_list)],
                            "predicted_binary": binary_preds[: len(mask_list)],
                        }
                    )

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                continue

        # Calculate comprehensive metrics
        metrics = self._calculate_metrics(
            all_true_labels, all_predictions, all_pred_scores
        )

        return {
            "metrics": metrics,
            "detailed_results": detailed_results,
            "summary": {
                "total_samples": len(all_true_labels),
                "positive_samples": sum(all_true_labels),
                "threshold": self.config.threshold,
            },
        }

    def _calculate_metrics(
        self, y_true: List[int], y_pred: List[int], y_scores: List[float]
    ) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_scores = np.array(y_scores)

        # Basic metrics
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        # Calculate metrics with zero division handling
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

        # Balanced accuracy
        balanced_accuracy = (recall + specificity) / 2

        # AUC-ROC (if we have positive and negative samples)
        try:
            auc_roc = (
                roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0
            )
        except Exception:
            auc_roc = 0.0

        # Matthews Correlation Coefficient
        mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else 0.0

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1_score": float(f1),
            "balanced_accuracy": float(balanced_accuracy),
            "auc_roc": float(auc_roc),
            "mcc": float(mcc),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        }

    def save_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results to files."""
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save main results
        results_path = results_dir / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save metrics summary
        metrics_path = results_dir / "metrics_summary.json"
        with open(metrics_path, "w") as f:
            json.dump(
                {"metrics": results["metrics"], "summary": results["summary"]},
                f,
                indent=2,
                default=str,
            )

        logger.info(f"Results saved to {results_dir}")

        # Print metrics summary
        logger.info("=== Evaluation Results ===")
        for metric, value in results["metrics"].items():
            if isinstance(value, float):
                logger.info(f"{metric.upper()}: {value:.4f}")
            else:
                logger.info(f"{metric.upper()}: {value}")


def main():
    """Main evaluation function."""
    config = EvaluationConfig()

    # Load test dataset
    logger.info("Loading test dataset...")
    data = load_dataset(split="test")

    # Initialize evaluator
    evaluator = ModelEvaluator(config)

    # Run evaluation
    results = evaluator.evaluate_dataset(data)

    # Save results
    evaluator.save_results(results)


if __name__ == "__main__":
    main()
