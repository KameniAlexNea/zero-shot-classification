import os

os.environ["WANDB_PROJECT"] = "zero-shot-classification"
os.environ["WANDB_WATCH"] = "none"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import Dataset
from loguru import logger
from sentence_transformers.cross_encoder import CrossEncoder
from tqdm import tqdm

from evaluation.metrics import compute_topk_metrics as compute_metrics
from gliznet import LabelName
from gliznet.evaluation_ds import ds_mapping


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""

    model_path: str = "results/best_model/model"
    device: str = "auto"
    batch_size: int = 64
    max_labels: int = 20
    threshold: float = 0.5
    results_dir: str = "results/evaluation"
    activation: str = "softmax"
    entailment: int = -1


class ModelEvaluator:
    """Handles model evaluation with comprehensive metrics."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = self._get_device()
        self.model = CrossEncoder(
            model_name_or_path=self.config.model_path,
            device=self.device,
        )

    def _get_device(self) -> torch.device:
        """Get the appropriate device for inference."""
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config.device)

    def predict_batch(self, inputs: dict[str, list[str]]) -> List[List[float]]:
        """Make predictions for a batch of inputs."""

        def activate(logits: torch.Tensor) -> torch.Tensor:
            """Apply activation function to logits."""
            if self.config.activation == "softmax":
                return logits.softmax(dim=-1)
            elif self.config.activation == "sigmoid":
                return logits.sigmoid()
            else:
                raise ValueError(
                    f"Unsupported activation function: {self.config.activation}"
                )

        with torch.no_grad():
            inputs_gpu = [
                (text, label)
                for text, labels in zip(inputs["text"], inputs[LabelName.ltext])
                for label in labels
            ]
            predictions = self.model.predict(
                inputs_gpu,
                activation_fn=torch.nn.Identity(),
                convert_to_tensor=True,
                convert_to_numpy=False,
            )
            i = 0
            labels: list[list[str]] = inputs[LabelName.ltext]
            all_predictions = []
            pos = 0
            while i < len(predictions):
                all_predictions.append(
                    (
                        activate(predictions[i : i + len(labels[pos])])
                        if predictions.ndim == 1 or predictions.shape[1] == 1
                        else activate(
                            predictions[
                                i : i + len(labels[pos]), self.config.entailment
                            ]
                        )
                    )
                    .cpu()
                    .numpy()
                )
                i += len(labels[pos])
                pos += 1

        return all_predictions

    def evaluate_dataset(self, dataset: Dataset) -> Dict[str, Any]:
        """Evaluate the model on the given dataset."""
        logger.info("Starting evaluation...")

        all_predictions = []
        all_true_labels = []

        for batch in tqdm(
            dataset.iter(batch_size=self.config.batch_size), desc="Evaluating"
        ):
            try:
                labels = batch.pop(LabelName.lint)
                logits = self.predict_batch(batch)

                all_predictions.append(logits)
                all_true_labels.append([np.array(label) for label in labels])

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                raise e

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
        if self.config.results_dir is not None:
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


def get_args():
    args = ArgumentParser(description="Evaluate GliZNet model on test dataset.")
    args.add_argument(
        "--model_path",
        type=str,
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
        default=None,
        help="Directory to save evaluation results.",
    )
    args.add_argument(
        "--activation",
        type=str,
        default="softmax",
        help="Activation function to use for model outputs.",
    )
    args.add_argument(
        "--data",
        type=str,
        default="agnews",
        help="Dataset to evaluate on (agnews or imdb).",
    )
    args.add_argument(
        "--entailment",
        type=int,
        default=-1,
        help="Use entailment labels (1) or not (0). Default is -1",
    )
    args.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation.",
    )
    args.add_argument(
        "--device_pos",
        type=int,
        default=0,
        help="Batch size for evaluation.",
    )
    args = args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_pos)
    return args


args = get_args()


def main():
    """Main evaluation function."""
    config = EvaluationConfig(
        model_path=args.model_path,
        threshold=args.threshold,
        results_dir=args.results_dir,
        activation=args.activation,
        entailment=args.entailment,
        batch_size=args.batch_size,
    )

    # Initialize evaluator
    evaluator = ModelEvaluator(config)

    logger.info("Loading test dataset...")
    if args.data not in ds_mapping:
        raise ValueError("Invalid dataset specified. Choose " + str(ds_mapping.keys()))
    logger.info(f"Using {args.data} dataset for evaluation.")
    data = ds_mapping[args.data]()

    # Run evaluation
    results = evaluator.evaluate_dataset(data)

    # Save results
    evaluator.save_results(results)


if __name__ == "__main__":
    main()
