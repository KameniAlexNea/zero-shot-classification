import os
import importlib
import json
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import torch
from datasets import Dataset
from loguru import logger
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from sklearn.linear_model import LogisticRegression

from gliznet.data import add_tokenized_function
from evaluation.mteb_ds import ds_mapping as mteb_ds_mapping
from gliznet.evaluation_ds import ds_mapping
from gliznet.model import create_gli_znet_for_sequence_classification
from gliznet.tokenizer import GliZNETTokenizer

# Combine both dataset mappings
ds_mapping = {**ds_mapping, **mteb_ds_mapping}


def get_transformers_class(class_name):
    transformers_module = importlib.import_module("transformers")
    return getattr(transformers_module, class_name)


@dataclass
class MTEBEvaluationConfig:
    """Configuration for MTEB-style evaluation."""
    model_path: str = "results/best_model/model"
    model_class: str = "BertPreTrainedModel"
    device: str = "auto"
    batch_size: int = 64
    results_dir: str = "results/mteb_evaluation"
    use_fast_tokenizer: bool = True
    k: int = 5  # Number of neighbors for KNN
    limit: Optional[int] = None  # Limit samples for testing
    classifier_type: str = "knn"  # "knn" or "logreg"
    max_iter: int = 100  # For logistic regression


class MTEBModelWrapper:
    """Wrapper to make GliZNet compatible with MTEB-style encoding."""
    
    def __init__(self, model, tokenizer, device, batch_size=32):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
    
    def encode(self, sentences: List[str], task_name: Optional[str] = None, **kwargs) -> np.ndarray:
        """Encode sentences using the model's encode method."""
        batch_size = kwargs.get("batch_size", self.batch_size)
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding sentences"):
                batch_sentences = sentences[i:i + batch_size]
                
                # Tokenize the batch
                encoded = self.tokenizer(
                    batch_sentences,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512
                )
                
                # Move to device
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                
                # Get embeddings using the model's encode method
                batch_embeddings = self.model.encode(input_ids, attention_mask)
                
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)


class kNNClassificationEvaluator:
    """KNN-based classification evaluator similar to MTEB."""
    
    def __init__(
        self,
        sentences_train: List[str],
        y_train: List[int],
        sentences_test: List[str],
        y_test: List[int],
        task_name: Optional[str] = None,
        k: int = 1,
        encode_kwargs: Dict[str, Any] = None,
        limit: Optional[int] = None,
    ):
        if encode_kwargs is None:
            encode_kwargs = {}
            
        if limit is not None:
            warnings.warn(
                "Limiting the number of samples with `limit` for evaluation.",
                UserWarning,
            )
            sentences_train = sentences_train[:limit]
            y_train = y_train[:limit]
            sentences_test = sentences_test[:limit]
            y_test = y_test[:limit]
            
        self.sentences_train = sentences_train
        self.y_train = y_train
        self.sentences_test = sentences_test
        self.y_test = y_test
        self.task_name = task_name
        self.encode_kwargs = encode_kwargs
        self.k = k
        
        if "batch_size" not in self.encode_kwargs:
            self.encode_kwargs["batch_size"] = 32

    def __call__(self, model_wrapper: MTEBModelWrapper, test_cache=None) -> Tuple[Dict[str, float], np.ndarray]:
        """Run KNN evaluation."""
        scores = {}
        max_accuracy = 0
        max_f1 = 0
        max_ap = 0
        
        logger.info("Encoding training sentences...")
        X_train = model_wrapper.encode(
            self.sentences_train,
            task_name=self.task_name,
            **self.encode_kwargs,
        )
        
        logger.info("Encoding test sentences...")
        if test_cache is None:
            X_test = model_wrapper.encode(
                self.sentences_test,
                task_name=self.task_name,
                **self.encode_kwargs,
            )
            test_cache = X_test
        else:
            X_test = test_cache
            
        # Evaluate with different distance metrics
        for metric in ["cosine", "euclidean"]:
            logger.info(f"Running KNN with {metric} distance...")
            knn = KNeighborsClassifier(n_neighbors=self.k, n_jobs=-1, metric=metric)
            knn.fit(X_train, self.y_train)
            y_pred = knn.predict(X_test)
            
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average="macro")
            
            scores[f"accuracy_{metric}"] = accuracy
            scores[f"f1_{metric}"] = f1
            
            max_accuracy = max(max_accuracy, accuracy)
            max_f1 = max(max_f1, f1)
            
            # Binary classification metrics
            if len(np.unique(self.y_train)) == 2:
                ap = average_precision_score(self.y_test, y_pred)
                scores[f"ap_{metric}"] = ap
                max_ap = max(max_ap, ap)
                
        scores["accuracy"] = max_accuracy
        scores["f1"] = max_f1
        if len(np.unique(self.y_train)) == 2:
            scores["ap"] = max_ap
            
        return scores, test_cache


class LogRegClassificationEvaluator:
    """Logistic regression-based classification evaluator similar to MTEB."""
    
    def __init__(
        self,
        sentences_train: List[str],
        y_train: List[int],
        sentences_test: List[str],
        y_test: List[int],
        task_name: Optional[str] = None,
        max_iter: int = 100,
        encode_kwargs: Dict[str, Any] = None,
        limit: Optional[int] = None,
    ):
        if encode_kwargs is None:
            encode_kwargs = {}
            
        if limit is not None:
            warnings.warn(
                "Limiting the number of samples with `limit` for evaluation.",
                UserWarning,
            )
            sentences_train = sentences_train[:limit]
            y_train = y_train[:limit]
            sentences_test = sentences_test[:limit]
            y_test = y_test[:limit]
            
        self.sentences_train = sentences_train
        self.y_train = y_train
        self.sentences_test = sentences_test
        self.y_test = y_test
        self.task_name = task_name
        self.encode_kwargs = encode_kwargs
        self.max_iter = max_iter
        
        if "batch_size" not in self.encode_kwargs:
            self.encode_kwargs["batch_size"] = 32

    def __call__(self, model_wrapper: MTEBModelWrapper, test_cache=None) -> Tuple[Dict[str, float], np.ndarray]:
        """Run logistic regression evaluation."""
        scores = {}
        
        clf = LogisticRegression(
            random_state=42,
            n_jobs=-1,
            max_iter=self.max_iter,
        )
        
        logger.info("Encoding training sentences...")
        X_train = model_wrapper.encode(
            self.sentences_train,
            task_name=self.task_name,
            **self.encode_kwargs,
        )
        
        logger.info("Encoding test sentences...")
        if test_cache is None:
            X_test = model_wrapper.encode(
                self.sentences_test,
                task_name=self.task_name,
                **self.encode_kwargs,
            )
            test_cache = X_test
        else:
            X_test = test_cache
            
        logger.info("Fitting logistic regression classifier...")
        clf.fit(X_train, self.y_train)
        
        logger.info("Evaluating...")
        y_pred = clf.predict(X_test)
        
        scores["accuracy"] = accuracy_score(self.y_test, y_pred)
        scores["f1"] = f1_score(self.y_test, y_pred, average="macro")
        scores["f1_weighted"] = f1_score(self.y_test, y_pred, average="weighted")
        
        # Binary classification metrics
        if len(np.unique(self.y_train)) == 2:
            scores["ap"] = average_precision_score(self.y_test, y_pred, average="macro")
            scores["ap_weighted"] = average_precision_score(self.y_test, y_pred, average="weighted")
            
        return scores, test_cache


class MTEBStyleEvaluator:
    """Main evaluator that uses MTEB-style evaluation with embeddings and KNN/LogReg."""
    
    def __init__(self, config: MTEBEvaluationConfig):
        self.config = config
        self.device = self._get_device()
        self.model, self.tokenizer = self._load_models()
        self.model_wrapper = MTEBModelWrapper(
            self.model, self.tokenizer, self.device, config.batch_size
        )

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
            ).from_pretrained(self.config.model_path)
            
            model.to(self.device)
            model.eval()

            logger.info("Model loaded successfully")
            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def prepare_data_for_classification(self, dataset: Dataset) -> Tuple[List[str], List[str], List[int], List[int]]:
        """Prepare the dataset for MTEB-style classification."""
        sentences_train = []
        sentences_test = []
        y_train = []
        y_test = []
        
        # Convert dataset to classification format
        texts = []
        labels = []
        
        for item in dataset:
            text = item["text"]
            label_texts = item["ltext"]
            label_ints = item["lint"]
            
            # Find the positive label (first True in lint)
            positive_idx = next((i for i, val in enumerate(label_ints) if val), 0)
            label_text = label_texts[positive_idx]
            
            texts.append(text)
            labels.append(label_text)
        
        # Create label mapping
        unique_labels = list(set(labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        y_numeric = [label_to_idx[label] for label in labels]
        
        # Split into train/test (using first 80% for train, rest for test)
        split_idx = int(0.8 * len(texts))
        
        sentences_train = texts[:split_idx]
        sentences_test = texts[split_idx:]
        y_train = y_numeric[:split_idx]
        y_test = y_numeric[split_idx:]
        
        logger.info(f"Prepared {len(sentences_train)} training samples and {len(sentences_test)} test samples")
        logger.info(f"Number of unique labels: {len(unique_labels)}")
        
        return sentences_train, sentences_test, y_train, y_test

    def evaluate_dataset(self, dataset: Dataset) -> Dict[str, Any]:
        """Evaluate the model on the given dataset using MTEB-style evaluation."""
        logger.info("Starting MTEB-style evaluation...")
        
        # Prepare data for classification
        sentences_train, sentences_test, y_train, y_test = self.prepare_data_for_classification(dataset)
        
        # Apply limit if specified
        if self.config.limit is not None:
            sentences_train = sentences_train[:self.config.limit]
            y_train = y_train[:self.config.limit]
            sentences_test = sentences_test[:self.config.limit]
            y_test = y_test[:self.config.limit]
        
        # Choose evaluator based on config
        if self.config.classifier_type == "knn":
            evaluator = kNNClassificationEvaluator(
                sentences_train=sentences_train,
                y_train=y_train,
                sentences_test=sentences_test,
                y_test=y_test,
                k=self.config.k,
                encode_kwargs={"batch_size": self.config.batch_size},
                limit=None,  # Already applied above
            )
        else:  # logreg
            evaluator = LogRegClassificationEvaluator(
                sentences_train=sentences_train,
                y_train=y_train,
                sentences_test=sentences_test,
                y_test=y_test,
                max_iter=self.config.max_iter,
                encode_kwargs={"batch_size": self.config.batch_size},
                limit=None,  # Already applied above
            )
        
        # Run evaluation
        metrics, test_cache = evaluator(self.model_wrapper)
        
        return {
            "metrics": metrics,
            "classifier_type": self.config.classifier_type,
            "summary": {
                "num_train_samples": len(sentences_train),
                "num_test_samples": len(sentences_test),
                "num_labels": len(set(y_train + y_test)),
                "k": self.config.k if self.config.classifier_type == "knn" else None,
            },
        }

    def save_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results to files."""
        if self.config.results_dir is not None:
            results_dir = Path(self.config.results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)

            # Save main results
            results_path = results_dir / "mteb_evaluation_results.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

            # Save metrics summary
            metrics_path = results_dir / "mteb_metrics_summary.json"
            with open(metrics_path, "w") as f:
                json.dump(
                    {"metrics": results["metrics"], "summary": results["summary"]},
                    f,
                    indent=2,
                    default=str,
                )

            logger.info(f"Results saved to {results_dir}")

        # Print metrics summary
        logger.info("=== MTEB-Style Evaluation Results ===")
        for metric, value in results["metrics"].items():
            if isinstance(value, float):
                logger.info(f"{metric.upper()}: {value:.4f}")
            else:
                logger.info(f"{metric.upper()}: {value}")


def get_args():
    parser = ArgumentParser(description="Evaluate GliZNet model using MTEB-style evaluation.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model directory.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/mteb_evaluation",
        help="Directory to save evaluation results.",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="DebertaV2PreTrainedModel",
        help="Model class to use",
    )
    parser.add_argument(
        "--use_fast_tokenizer",
        action="store_true",
        help="Use fast tokenizer if available.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="events_biotech",
        help="Dataset to evaluate on.",
    )
    parser.add_argument(
        "--device_pos",
        type=int,
        default=0,
        help="CUDA device position.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of neighbors for KNN classification.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples for testing.",
    )
    parser.add_argument(
        "--classifier_type",
        type=str,
        default="knn",
        choices=["knn", "logreg"],
        help="Type of classifier to use (knn or logreg).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for encoding.",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=100,
        help="Maximum iterations for logistic regression.",
    )
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_pos)
    return args


def main():
    """Main evaluation function."""
    args = get_args()
    
    config = MTEBEvaluationConfig(
        model_path=args.model_path,
        results_dir=args.results_dir,
        model_class=args.model_class,
        use_fast_tokenizer=args.use_fast_tokenizer,
        k=args.k,
        limit=args.limit,
        classifier_type=args.classifier_type,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
    )

    # Initialize evaluator
    evaluator = MTEBStyleEvaluator(config)

    # Load test dataset
    logger.info("Loading dataset...")
    if args.data not in ds_mapping:
        raise ValueError(f"Invalid dataset specified. Choose from: {list(ds_mapping.keys())}")
    
    logger.info(f"Using {args.data} dataset for evaluation.")
    dataset = ds_mapping[args.data]()

    
    dataset = add_tokenized_function(
        hf_dataset=dataset,
        tokenizer=evaluator.tokenizer,
        max_labels=100,
        shuffle_labels=False,
    )

    # Run evaluation
    results = evaluator.evaluate_dataset(dataset)

    # Save results
    evaluator.save_results(results)


if __name__ == "__main__":
    main()
