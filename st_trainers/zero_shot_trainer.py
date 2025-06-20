import os

from dataclasses import dataclass, field
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
from transformers import HfArgumentParser
from tqdm import tqdm


@dataclass
class ModelArgs:
    model_name: str = field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        metadata={"help": "Base model name or path"},
    )
    dataset_name: str = field(
        default="alexneakameni/ZSHOT-HARDSET",
        metadata={"help": "Hugging Face dataset name"},
    )
    train_split: str = field(
        default="couplet", 
        metadata={"help": "Dataset split for training"}
    )
    eval_split: str = field(
        default="triplet", 
        metadata={"help": "Dataset split for evaluation"}
    )
    mini_batch_size: int = field(
        default=32, 
        metadata={"help": "Mini batch size for CachedMultipleNegativesRankingLoss"}
    )
    scale: float = field(
        default=20.0, 
        metadata={"help": "Scale for similarity function"}
    )
    margin: float = field(
        default=0.1, 
        metadata={"help": "Margin for triplet evaluation"}
    )


class ZeroShotTrainer:
    def __init__(self, model_args: ModelArgs):
        self.model_args = model_args
        self.all_labels = set()
        self.model = None

        logger.info(
            f"Initialized ZeroShotTrainer with SentenceTransformer: {model_args.model_name}"
        )
        logger.info(
            f"Will use dataset: {model_args.dataset_name}, train split: {model_args.train_split}, eval split: {model_args.eval_split}"
        )

    def load_training_dataset(self) -> Tuple[Dataset, Dataset, Set[str]]:
        """Load couplet dataset for training."""
        logger.info(
            f"Loading TRAINING dataset: {self.model_args.dataset_name}/{self.model_args.train_split}"
        )

        try:
            # Load the couplet dataset for training
            dataset = load_dataset(self.model_args.dataset_name, self.model_args.train_split)
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
            f"Loading EVALUATION dataset: {self.model_args.dataset_name}/{self.model_args.eval_split}"
        )

        try:
            # Load the triplet dataset for evaluation
            dataset = load_dataset(self.model_args.dataset_name, self.model_args.eval_split)
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

    def train(self, training_args: SentenceTransformerTrainingArguments):
        """Complete training pipeline using couplet for training and triplet for evaluation."""

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
            model_name_or_path=self.model_args.model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Choose appropriate loss function
        logger.info("Using CachedMultipleNegativesRankingLoss...")
        train_loss = losses.CachedMultipleNegativesRankingLoss(
            model=self.model,
            scale=self.model_args.scale,
            mini_batch_size=self.model_args.mini_batch_size,
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
            margin=self.model_args.margin,
            name="zero_shot_triplet_eval",
            batch_size=training_args.per_device_eval_batch_size,
            show_progress_bar=False,
        )

        # Initialize trainer
        logger.info("Initializing SentenceTransformerTrainer...")
        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            loss=train_loss,
            evaluator=evaluator,
        )

        # Train model
        logger.info("Starting training...")
        logger.info("Training setup:")
        logger.info(
            f"  • Training data: {self.model_args.dataset_name}/{self.model_args.train_split}"
        )
        logger.info(
            f"  • Evaluation data: {self.model_args.dataset_name}/{self.model_args.eval_split}"
        )
        logger.info(f"  • Loss: {type(train_loss).__name__}")
        logger.info(f"  • Evaluator: TripletEvaluator with {len(anchors)} triplets")
        logger.info(
            "🎯 Key insight: Training on couplets, evaluating with triplet hard negatives!"
        )

        trainer.train()

        # Save model and metadata
        os.makedirs(training_args.output_dir, exist_ok=True)
        self.model.save(training_args.output_dir)

        # Save dataset info
        dataset_info = {
            "dataset_name": self.model_args.dataset_name,
            "train_split": self.model_args.train_split,
            "eval_split": self.model_args.eval_split,
            "train_labels": list(train_labels),
            "eval_labels": list(eval_labels),
            "num_train_samples": len(train_hf),
            "num_eval_samples": len(eval_hf),
            "train_pairs": len(train_dataset),
            "eval_triplets": len(anchors),
            "has_hard_negatives_eval": True,
            "total_unique_labels": len(train_labels | eval_labels),
        }

        with open(os.path.join(training_args.output_dir, "dataset_info.json"), "w") as f:
            json.dump(dataset_info, f, indent=2)

        # Save config
        with open(os.path.join(training_args.output_dir, "training_config.json"), "w") as f:
            config_dict = {**self.model_args.__dict__, **training_args.to_dict()}
            json.dump(config_dict, f, indent=2)

        # Final evaluation
        logger.info("Performing final triplet evaluation with hard negatives...")
        final_score = evaluator(
            self.model, output_path=os.path.join(training_args.output_dir, "final_eval_triplet.csv")
        )

        logger.success(
            f"Final triplet accuracy with hard negatives: {final_score[evaluator.primary_metric]:.4f}"
        )


        logger.success(f"Training completed! Model saved to {training_args.output_dir}")
        return self.model


def main():
    parser = HfArgumentParser((ModelArgs, SentenceTransformerTrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    # Set device
    device = (
        "cuda" if torch.cuda.is_available() and not training_args.no_cuda else "cpu"
    )
    logger.info(f"Using device: {device}")

    # Create output directory
    os.makedirs(training_args.output_dir, exist_ok=True)

    trainer = ZeroShotTrainer(model_args)
    trainer.train(training_args)


if __name__ == "__main__":
    main()