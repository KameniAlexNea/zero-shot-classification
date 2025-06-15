#!/usr/bin/env python3
"""
Inference script for FZeroNet - Zero-shot Classification Model
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from loguru import logger

from gliznet.model import FZeroNet
from gliznet.tokenizer import ZeroShotClassificationTokenizer


class FZeroNetInference:
    """
    Inference wrapper for FZeroNet model.
    """

    def __init__(
        self,
        model_path: str,
        model_name: str = "bert-base-uncased",
        device: str = "auto",
    ):
        """
        Initialize inference model.

        Args:
            model_path: Path to saved model checkpoint
            model_name: HuggingFace model name used for training
            device: Device to use for inference
        """
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Loading model from {model_path} on {self.device}")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Initialize tokenizer
        self.tokenizer = ZeroShotClassificationTokenizer(model_name=model_name)

        # Initialize model
        self.model = FZeroNet(model_name=model_name)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        logger.info("Model loaded successfully")

    def classify(
        self,
        text: str,
        positive_labels: List[str],
        negative_labels: List[str],
        threshold: float = 0.5,
        return_all_scores: bool = False,
    ) -> Dict:
        """
        Classify a single text with given labels.

        Args:
            text: Input text to classify
            positive_labels: List of positive class labels to consider
            negative_labels: List of negative class labels to consider
            threshold: Classification threshold
            return_all_scores: Whether to return scores for all labels

        Returns:
            Classification results
        """
        all_labels = positive_labels + negative_labels

        # Tokenize
        inputs = self.tokenizer.tokenize_example(text, all_labels, return_tensors="pt")

        # Move to device and add batch dimension if needed
        for key in ["input_ids", "attention_mask", "label_mask"]:
            if inputs[key].dim() == 1:
                inputs[key] = inputs[key].unsqueeze(0)
            inputs[key] = inputs[key].to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                label_mask=inputs["label_mask"],
            )

            logits = outputs["logits"]
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
            predictions = (probabilities > threshold).astype(int)

        # Split results
        num_positive = len(positive_labels)
        positive_probs = probabilities[:num_positive]
        negative_probs = probabilities[num_positive:]
        positive_preds = predictions[:num_positive]
        negative_preds = predictions[num_positive:]

        # Build results
        results = {
            "text": text,
            "predicted_positive_labels": [
                positive_labels[i] for i, pred in enumerate(positive_preds) if pred == 1
            ],
            "predicted_negative_labels": [
                negative_labels[i] for i, pred in enumerate(negative_preds) if pred == 1
            ],
            "positive_scores": {
                label: float(score)
                for label, score in zip(positive_labels, positive_probs)
            },
            "negative_scores": {
                label: float(score)
                for label, score in zip(negative_labels, negative_probs)
            },
            "threshold": threshold,
        }

        if return_all_scores:
            results["all_scores"] = {
                label: float(score) for label, score in zip(all_labels, probabilities)
            }
            results["raw_logits"] = logits.cpu().numpy().tolist()

        return results

    def classify_batch(
        self,
        texts: List[str],
        positive_labels: List[str],
        negative_labels: List[str],
        threshold: float = 0.5,
        batch_size: int = 8,
    ) -> List[Dict]:
        """
        Classify a batch of texts.

        Args:
            texts: List of input texts
            positive_labels: List of positive class labels
            negative_labels: List of negative class labels
            threshold: Classification threshold
            batch_size: Batch size for processing

        Returns:
            List of classification results
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            for text in batch_texts:
                result = self.classify(
                    text, positive_labels, negative_labels, threshold
                )
                results.append(result)

        return results


def interactive_mode(inference_model: FZeroNetInference):
    """
    Interactive mode for testing the model.
    """
    print("\n" + "=" * 60)
    print("FZeroNet Interactive Classification")
    print("=" * 60)
    print("Type 'quit' to exit")

    while True:
        print("\n" + "-" * 40)

        # Get text input
        text = input("Enter text to classify: ").strip()
        if text.lower() == "quit":
            break

        if not text:
            print("Please enter some text.")
            continue

        # Get positive labels
        positive_input = input("Enter positive labels (comma-separated): ").strip()
        if not positive_input:
            print("Please enter at least one positive label.")
            continue
        positive_labels = [label.strip() for label in positive_input.split(",")]

        # Get negative labels
        negative_input = input("Enter negative labels (comma-separated): ").strip()
        if not negative_input:
            print("Please enter at least one negative label.")
            continue
        negative_labels = [label.strip() for label in negative_input.split(",")]

        # Get threshold
        threshold_input = input("Enter threshold (default 0.5): ").strip()
        try:
            threshold = float(threshold_input) if threshold_input else 0.5
        except ValueError:
            threshold = 0.5
            print(f"Invalid threshold, using default: {threshold}")

        # Classify
        try:
            result = inference_model.classify(
                text,
                positive_labels,
                negative_labels,
                threshold,
                return_all_scores=True,
            )

            # Display results
            print("\nResults:")
            print(f"Text: {text}")
            print(f"Threshold: {threshold}")

            print(f"\nPredicted Positive Labels: {result['predicted_positive_labels']}")
            print(f"Predicted Negative Labels: {result['predicted_negative_labels']}")

            print("\nPositive Label Scores:")
            for label, score in result["positive_scores"].items():
                print(f"  {label}: {score:.3f}")

            print("\nNegative Label Scores:")
            for label, score in result["negative_scores"].items():
                print(f"  {label}: {score:.3f}")

        except Exception as e:
            print(f"Error during classification: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run inference with FZeroNet")
    parser.add_argument(
        "--model_path", required=True, help="Path to saved model checkpoint"
    )
    parser.add_argument(
        "--model_name", default="bert-base-uncased", help="HuggingFace model name"
    )
    parser.add_argument(
        "--device", default="auto", help="Device to use (auto, cpu, cuda)"
    )
    parser.add_argument("--text", help="Text to classify")
    parser.add_argument("--positive_labels", help="Comma-separated positive labels")
    parser.add_argument("--negative_labels", help="Comma-separated negative labels")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Classification threshold"
    )
    parser.add_argument("--input_file", help="JSON file with classification tasks")
    parser.add_argument("--output_file", help="Output file for results")
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )

    args = parser.parse_args()

    # Check if model file exists
    if not Path(args.model_path).exists():
        logger.error(f"Model file not found: {args.model_path}")
        return

    # Initialize inference model
    try:
        inference_model = FZeroNetInference(
            model_path=args.model_path, model_name=args.model_name, device=args.device
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Interactive mode
    if args.interactive:
        interactive_mode(inference_model)
        return

    # Batch processing from file
    if args.input_file:
        if not Path(args.input_file).exists():
            logger.error(f"Input file not found: {args.input_file}")
            return

        logger.info(f"Processing input file: {args.input_file}")

        with open(args.input_file, "r") as f:
            tasks = json.load(f)

        results = []
        for task in tasks:
            result = inference_model.classify(
                text=task["text"],
                positive_labels=task["positive_labels"],
                negative_labels=task["negative_labels"],
                threshold=args.threshold,
                return_all_scores=True,
            )
            results.append(result)

        # Save results
        if args.output_file:
            with open(args.output_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output_file}")
        else:
            print(json.dumps(results, indent=2))

        return

    # Single classification
    if args.text and args.positive_labels and args.negative_labels:
        positive_labels = [label.strip() for label in args.positive_labels.split(",")]
        negative_labels = [label.strip() for label in args.negative_labels.split(",")]

        result = inference_model.classify(
            text=args.text,
            positive_labels=positive_labels,
            negative_labels=negative_labels,
            threshold=args.threshold,
            return_all_scores=True,
        )

        print(json.dumps(result, indent=2))
        return

    # If no specific mode is chosen, show help
    parser.print_help()


if __name__ == "__main__":
    main()
