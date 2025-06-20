import os

os.environ["WANDB_PROJECT"] = "gliznet"
os.environ["WANDB_WATCH"] = "none"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from dataclasses import dataclass
from typing import List, Set, Tuple
import json
import torch
from datasets import Dataset, load_dataset
from loguru import logger
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    losses,
    util,
    SentenceTransformerTrainingArguments
)
from sentence_transformers.evaluation import TripletEvaluator
from tqdm import tqdm

import wandb


@dataclass
class TrainingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    mini_batch_size: int = 32  # For CachedMultipleNegativesRankingLoss
    learning_rate: float = 2e-5
    num_epochs: int = 5
    max_length: int = 512
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    scale: float = 20.0  # Scale for similarity function
    evaluation_steps: int = 1000
    dataset_name: str = "alexneakameni/ZSHOT-HARDSET"
    train_split: str = "couplet"  # For training
    eval_split: str = "triplet"  # For evaluation


class ZeroShotTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.all_labels = set()
        self.model = None

        logger.info(
            f"Initialized ZeroShotTrainer with SentenceTransformer: {config.model_name}"
        )
        logger.info(
            f"Will use dataset: {config.dataset_name}, train split: {config.train_split}, eval split: {config.eval_split}"
        )

    def load_training_dataset(self) -> Tuple[Dataset, Dataset, Set[str]]:
        """Load couplet dataset for training."""
        logger.info(
            f"Loading TRAINING dataset: {self.config.dataset_name}/{self.config.train_split}"
        )

        try:
            # Load the couplet dataset for training
            dataset = load_dataset(self.config.dataset_name, self.config.train_split)
            train_dataset = dataset["train"]
            test_dataset = dataset["test"]

            logger.success("Loaded training dataset successfully:")
            logger.info(f"  Train samples: {len(train_dataset)}")
            logger.info(f"  Test samples: {len(test_dataset)}")

            # Extract all unique labels from training data
            train_labels = set()

            logger.info("Analyzing training label distribution...")
            for item in tqdm(train_dataset, desc="Processing train labels"):
                for label in item["labels"]:
                    train_labels.add(label)

            logger.info("Training dataset statistics:")
            logger.info(f"  Train labels: {len(train_labels)}")

            return train_dataset, test_dataset, train_labels

        except Exception as e:
            logger.error(f"Failed to load training dataset from Hugging Face: {e}")
            raise

    def load_evaluation_dataset(self) -> Tuple[Dataset, Set[str]]:
        """Load triplet dataset for evaluation."""
        logger.info(
            f"Loading EVALUATION dataset: {self.config.dataset_name}/{self.config.eval_split}"
        )

        try:
            # Load the triplet dataset for evaluation
            dataset = load_dataset(self.config.dataset_name, self.config.eval_split)
            eval_dataset = dataset["test"]  # Use test split for evaluation

            logger.success("Loaded evaluation dataset successfully:")
            logger.info(f"  Eval samples: {len(eval_dataset)}")

            # Check if it has hard negatives
            sample = eval_dataset[0]
            has_not_labels = "not_labels" in sample

            if has_not_labels:
                logger.info("✓ Evaluation dataset contains hard negatives (not_labels)")
            else:
                logger.warning("✗ Evaluation dataset missing hard negatives")

            # Extract all unique labels from evaluation data
            eval_labels = set()

            logger.info("Analyzing evaluation label distribution...")
            for item in tqdm(eval_dataset, desc="Processing eval labels"):
                for label in item["labels"]:
                    eval_labels.add(label)
                if has_not_labels:
                    for label in item.get("not_labels", []):
                        eval_labels.add(label)

            logger.info("Evaluation dataset statistics:")
            logger.info(f"  Eval labels: {len(eval_labels)}")

            return eval_dataset, eval_labels

        except Exception as e:
            logger.error(f"Failed to load evaluation dataset from Hugging Face: {e}")
            raise

    def create_contrastive_dataset(self, hf_dataset: Dataset) -> Dataset:
        """Convert HF dataset to contrastive learning format."""
        logger.info("Creating pairs (anchor, positive)")
        dataset_dict = {"anchor": [], "positive": []}

        text_key = "text" if "text" in hf_dataset.column_names else "sentence"

        for item in tqdm(hf_dataset, desc="Creating pairs"):
            text = item[text_key]
            positive_labels = item["labels"]

            for pos_label in positive_labels:
                dataset_dict["anchor"].append(text)
                dataset_dict["positive"].append(pos_label)

        logger.info(f"Created {len(dataset_dict['anchor'])} pairs")
        return Dataset.from_dict(dataset_dict)

    def create_triplet_evaluation_data(
        self, eval_dataset: Dataset
    ) -> Tuple[List[str], List[str], List[str]]:
        """Create triplet evaluation data (anchors, positives, negatives) from dataset."""
        anchors = []
        positives = []
        negatives = []

        text_key = "text" if "text" in eval_dataset.column_names else "sentence"

        logger.info("Creating triplet evaluation data...")

        for item in tqdm(eval_dataset, desc="Creating triplet evaluation data"):
            text = item[text_key]
            positive_labels = item["labels"]
            negative_labels = item.get("not_labels", [])

            # Create all combinations of positive and negative labels for this text
            for pos_label in positive_labels:
                for neg_label in negative_labels:
                    anchors.append(text)
                    positives.append(pos_label)
                    negatives.append(neg_label)

        logger.info(f"Created {len(anchors)} triplets for evaluation")
        return anchors, positives, negatives

    def train(self, output_dir: str = "models/zero_shot_classifier"):
        """Complete training pipeline using couplet for training and triplet for evaluation."""

        # Initialize wandb
        wandb.init(
            project="zero-shot-classification",
            config=self.config.__dict__,
            name=f"zero-shot-train-{self.config.train_split}-eval-{self.config.eval_split}-{self.config.model_name.split('/')[-1]}",
        )

        # Load training dataset (couplet)
        train_hf, train_test_hf, train_labels = self.load_training_dataset()

        # Load evaluation dataset (triplet)
        eval_hf, eval_labels = self.load_evaluation_dataset()

        # Check for label overlap between train and eval
        label_overlap = train_labels & eval_labels
        if label_overlap:
            logger.warning(
                f"Found {len(label_overlap)} overlapping labels between train/eval"
            )
            logger.info(f"  Overlap examples: {list(label_overlap)[:10]}")
        else:
            logger.success("✓ No label overlap - true zero-shot setup!")

        # Convert training data to contrastive learning format
        train_dataset = self.create_contrastive_dataset(train_hf)

        logger.info("Training dataset created:")
        logger.info(f"  Train: {len(train_dataset)} pairs")

        # Initialize model
        logger.info("Initializing SentenceTransformer...")
        self.model = SentenceTransformer(
            model_name_or_path=self.config.model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Choose appropriate loss function
        logger.info("Using CachedMultipleNegativesRankingLoss...")
        train_loss = losses.CachedMultipleNegativesRankingLoss(
            model=self.model,
            scale=self.config.scale,
            mini_batch_size=self.config.mini_batch_size,
            show_progress_bar=False,
            similarity_fct=util.cos_sim,
        )

        # Create TripletEvaluator using triplet data with hard negatives
        logger.info(
            "Setting up TripletEvaluator with triplet data and hard negatives..."
        )
        anchors, positives, negatives = self.create_triplet_evaluation_data(eval_hf)

        evaluator = TripletEvaluator(
            anchors=anchors,
            positives=positives,
            negatives=negatives,
            main_similarity_function="cosine",
            margin=0.1,  # Margin for triplet evaluation
            name="zero_shot_triplet_eval",
            batch_size=self.config.batch_size,
            show_progress_bar=False,
        )

        # Training arguments
        args = {
            "output_dir": output_dir,
            "num_train_epochs": self.config.num_epochs,
            "per_device_train_batch_size": self.config.batch_size,
            "per_device_eval_batch_size": self.config.batch_size,
            "warmup_steps": self.config.warmup_steps,
            "weight_decay": self.config.weight_decay,
            "learning_rate": self.config.learning_rate,
            "logging_steps": 100,
            "eval_strategy": "steps",
            "eval_steps": self.config.evaluation_steps,
            "save_steps": self.config.evaluation_steps,
            "save_strategy": "steps",
            "load_best_model_at_end": True,
            "metric_for_best_model": "zero_shot_triplet_eval_cosine_accuracy",
            "greater_is_better": True,
            "report_to": "wandb",
            "run_name": f"zero-shot-train-{self.config.train_split}-eval-{self.config.eval_split}",
            "dataloader_num_workers": 4,
            "fp16": torch.cuda.is_available(),
        }

        # Initialize trainer
        logger.info("Initializing SentenceTransformerTrainer...")
        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=SentenceTransformerTrainingArguments(**args),
            train_dataset=train_dataset,
            loss=train_loss,
            evaluator=evaluator,
        )

        # Train model
        logger.info("Starting training...")
        logger.info("Training setup:")
        logger.info(
            f"  • Training data: {self.config.dataset_name}/{self.config.train_split}"
        )
        logger.info(
            f"  • Evaluation data: {self.config.dataset_name}/{self.config.eval_split}"
        )
        logger.info(f"  • Loss: {type(train_loss).__name__}")
        logger.info(f"  • Evaluator: TripletEvaluator with {len(anchors)} triplets")
        logger.info(
            "🎯 Key insight: Training on couplets, evaluating with triplet hard negatives!"
        )

        trainer.train()

        # Save model and metadata
        os.makedirs(output_dir, exist_ok=True)
        self.model.save(output_dir)

        # Save dataset info
        dataset_info = {
            "dataset_name": self.config.dataset_name,
            "train_split": self.config.train_split,
            "eval_split": self.config.eval_split,
            "train_labels": list(train_labels),
            "eval_labels": list(eval_labels),
            "num_train_samples": len(train_hf),
            "num_eval_samples": len(eval_hf),
            "train_pairs": len(train_dataset),
            "eval_triplets": len(anchors),
            "has_hard_negatives_eval": True,
            "total_unique_labels": len(train_labels | eval_labels),
        }

        with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
            json.dump(dataset_info, f, indent=2)

        # Save config
        with open(os.path.join(output_dir, "training_config.json"), "w") as f:
            json.dump(self.config.__dict__, f, indent=2)

        # Final evaluation
        logger.info("Performing final triplet evaluation with hard negatives...")
        final_score = evaluator(
            self.model, output_path=os.path.join(output_dir, "final_eval_triplet.csv")
        )

        logger.success(
            f"Final triplet accuracy with hard negatives: {final_score[evaluator.primary_metric]:.4f}"
        )

        wandb.log({"final_triplet_accuracy": final_score[evaluator.primary_metric]})
        wandb.finish()

        logger.success(f"Training completed! Model saved to {output_dir}")
        return self.model


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train zero-shot classification model")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="alexneakameni/ZSHOT-HARDSET",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--train-split", type=str, default="couplet", help="Dataset split for training"
    )
    parser.add_argument(
        "--eval-split", type=str, default="triplet", help="Dataset split for evaluation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/zero_shot_classifier",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base model name",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument(
        "--mini-batch-size",
        type=int,
        default=32,
        help="Mini batch size for gradient caching",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--scale", type=float, default=20.0, help="Scale for similarity function"
    )

    args = parser.parse_args()

    config = TrainingConfig(
        dataset_name=args.dataset_name,
        train_split=args.train_split,
        eval_split=args.eval_split,
        model_name=args.model_name,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        scale=args.scale,
    )

    trainer = ZeroShotTrainer(config)
    trainer.train(args.output_dir)


if __name__ == "__main__":
    main()
