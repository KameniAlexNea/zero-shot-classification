import importlib
import json
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import Dataset
from loguru import logger
from tqdm import tqdm

from gliznet.data import add_tokenized_function
from gliznet.evaluation_ds import (
    load_agnews_dataset,
    load_amazon_massive_intent,
    load_imdb_dataset,
)
from gliznet.metrics import compute_metrics
from gliznet.model import create_gli_znet_for_sequence_classification
from gliznet.tokenizer import GliZNETTokenizer

ds_mapping = {
    "agnews": load_agnews_dataset,
    "imdb": load_imdb_dataset,
    "amazon_massive_intent": load_amazon_massive_intent,
}


def get_transformers_class(class_name):
    transformers_module = importlib.import_module("transformers")
    return getattr(transformers_module, class_name)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""

    model_path: str = "results/best_model/model"
    model_class: str = "BertPreTrainedModel"
    device: str = "auto"
    batch_size: int = 64
    max_labels: int = 20
    threshold: float = 0.5
    results_dir: str = "results/evaluation_test"
    use_fast_tokenizer: bool = True
    activation: str = "softmax"


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

    def _load_models(self):
        """Load model and tokenizer from checkpoint."""
        logger.info(f"Loading model from {self.config.model_path} on {self.device}")

        try:
            tokenizer = GliZNETTokenizer.from_pretrained(
                self.config.model_path, use_fast=self.config.use_fast_tokenizer
            )

            model = create_gli_znet_for_sequence_classification(
                get_transformers_class(self.config.model_class)
            ).from_pretrained(
                self.config.model_path,
            )
            model.to(self.device)
            model.eval()

            logger.info("Model loaded successfully")
            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict_batch(self, inputs: dict[str, torch.Tensor]) -> List[List[float]]:
        """Make predictions for a batch of inputs."""

        with torch.no_grad():
            inputs_gpu = {k: v.to(self.device) for k, v in inputs.items()}
            predictions = self.model.predict(
                **inputs_gpu, activation_fn=self.config.activation
            )

        return predictions

    def evaluate_dataset(self, dataset: Dataset) -> Dict[str, Any]:
        """Evaluate the model on the given dataset."""
        logger.info("Starting evaluation...")

        all_predictions = []
        all_true_labels = []

        for batch in tqdm(
            dataset.iter(batch_size=self.config.batch_size), desc="Evaluating"
        ):
            try:
                labels = batch.pop("labels")
                logits = self.predict_batch(batch)

                all_predictions.append([np.array(logit) for logit in logits])
                all_true_labels.append([lab.numpy() for lab in labels])

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
    "--model_class",
    type=str,
    default="BertPreTrainedModel",
    help="Model class to use",
)
args.add_argument(
    "--use_fast_tokenizer",
    action="store_true",
    help="Use fast tokenizer if available.",
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
    "--max_labels",
    type=int,
    default=100,
    help="Maximum number of labels to consider for each example.",
)
args = args.parse_args()


def main():
    """Main evaluation function."""
    config = EvaluationConfig(
        model_path=args.model_path,
        threshold=args.threshold,
        results_dir=args.results_dir,
        model_class=args.model_class,
        use_fast_tokenizer=args.use_fast_tokenizer,
        activation=args.activation,
        max_labels=args.max_labels,
    )

    # Initialize evaluator
    evaluator = ModelEvaluator(config)

    # Load test dataset
    logger.info("Loading test dataset...")
    if args.data not in ds_mapping:
        raise ValueError("Invalid dataset specified. Choose " + str(ds_mapping.keys()))
    logger.info("Using IMDB dataset for evaluation." + args.data)
    data = ds_mapping[args.data]()

    data = add_tokenized_function(
        hf_dataset=data,
        tokenizer=evaluator.tokenizer,
        max_labels=config.max_labels,
        shuffle_labels=False,
    )

    # Run evaluation
    results = evaluator.evaluate_dataset(data)

    # Save results
    evaluator.save_results(results)


if __name__ == "__main__":
    main()
