import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from gliznet.config import LabelName
import numpy as np
import torch
from datasets import Dataset
from loguru import logger
from tqdm import tqdm

from gliznet.data import load_dataset
from gliznet.metrics import compute_metrics
from gliznet.predictor import ZeroShotClassificationPipeline


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""

    model_path: str = "alexneakameni/gliznet-ModernBERT-base"
    device: str = "auto"
    batch_size: int = 64
    threshold: float = 0.5
    results_dir: str = "results/evaluation_test"
    classification_type: str = "multi-label"


class ModelEvaluator:
    """Handles model evaluation with comprehensive metrics."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = self._get_device()
        self.pipeline = self._load_pipeline()

    def _get_device(self) -> torch.device:
        """Get the appropriate device for inference."""
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config.device)

    def _load_pipeline(self):
        """Load zero-shot classification pipeline."""
        logger.info(f"Loading pipeline from {self.config.model_path} on {self.device}")

        try:
            pipeline = ZeroShotClassificationPipeline.from_pretrained(
                self.config.model_path,
                classification_type=self.config.classification_type,
                device=str(self.device),
            )

            logger.info("Pipeline loaded successfully")
            return pipeline

        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise

    def predict_batch(
        self, texts: List[str], labels: List[List[str]]
    ) -> List[List[float]]:
        """Make predictions for a batch of inputs."""
        results = self.pipeline(texts, labels, threshold=None)

        # Convert pipeline output to logits format
        predictions = []
        for result in results:
            scores = [item["score"] for item in result]
            predictions.append(scores)

        return predictions

    def evaluate_dataset(self, dataset: Dataset) -> Dict[str, Any]:
        """Evaluate the model on the given dataset."""
        logger.info("Starting evaluation...")

        all_predictions = []
        all_true_labels = []

        # Process in batches
        for i in tqdm(
            range(0, len(dataset), self.config.batch_size),
            desc="Evaluating",
        ):
            try:
                batch = dataset[i : i + self.config.batch_size]
                texts = batch["text"]
                candidate_labels = batch[LabelName.ltext]
                true_labels = batch[LabelName.lint]

                # Get predictions
                logits = self.predict_batch(texts, candidate_labels)

                all_predictions.append([np.array(logit) for logit in logits])
                all_true_labels.append([np.array(lab) for lab in true_labels])

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                continue

        # Calculate comprehensive metrics
        metrics = compute_metrics((all_predictions, all_true_labels), True)

        return {
            "metrics": metrics,
            "detailed_results": {
                "predictions": [i.tolist() for j in all_predictions for i in j],
                "true_labels": [i.tolist() for j in all_true_labels for i in j],
            },
            "summary": {
                "threshold": self.config.threshold,
            },
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


args = ArgumentParser(description="Evaluate GliZNet model on test dataset.")
args.add_argument(
    "--model_path",
    type=str,
    default="alexneakameni/gliznet-ModernBERT-base",
    help="Path to the trained model directory.",
)
args.add_argument(
    "--threshold",
    type=float,
    default=0.5,
    help="Threshold for binary classification.",
)
args.add_argument(
    "--results_dir",
    type=str,
    default="results/evaluation",
    help="Directory to save evaluation results.",
)
args.add_argument(
    "--classification_type",
    type=str,
    default="multi-label",
    choices=["multi-label", "multi-class"],
    help="Classification type.",
)
args.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Batch size for evaluation.",
)
args = args.parse_args()


def main():
    """Main evaluation function."""
    config = EvaluationConfig(
        model_path=args.model_path,
        threshold=args.threshold,
        results_dir=args.results_dir,
        classification_type=args.classification_type,
        batch_size=args.batch_size,
    )

    # Initialize evaluator
    evaluator = ModelEvaluator(config)

    # Load test dataset (raw, not tokenized)
    logger.info("Loading test dataset...")
    data = load_dataset(split="test")

    # Run evaluation
    results = evaluator.evaluate_dataset(data)

    # Save results
    evaluator.save_results(results)


if __name__ == "__main__":
    main()
