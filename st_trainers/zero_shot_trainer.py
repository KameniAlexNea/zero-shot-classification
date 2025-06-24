import json
import os
from dataclasses import dataclass, field
from typing import List, Set, Tuple

import torch
from datasets import Dataset, load_dataset
from loguru import logger
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
    util,
)
from sentence_transformers.evaluation import TripletEvaluator
from tqdm import tqdm
from transformers import EarlyStoppingCallback, HfArgumentParser


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
        default="couplet", metadata={"help": "Dataset split for training"}
    )
    eval_split: str = field(
        default="triplet", metadata={"help": "Dataset split for evaluation"}
    )
    mini_batch_size: int = field(
        default=32,
        metadata={"help": "Mini batch size for CachedMultipleNegativesRankingLoss"},
    )
    scale: float = field(
        default=20.0, metadata={"help": "Scale for similarity function"}
    )
    margin: float = field(
        default=0.1, metadata={"help": "Margin for triplet evaluation"}
    )
    early_stopping_patience: int = field(
        default=3,
        metadata={"help": "Early stopping patience"},
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

    def load_dataset(
        self, split_name: str, split_type: str = "train"
    ) -> Tuple[Dataset, Set[str]]:
        """Load dataset and return standardized format with text, positive, negative (optional)."""
        logger.info(f"Loading dataset: {self.model_args.dataset_name}/{split_name}")

        try:
            # Load the dataset
            dataset = load_dataset(self.model_args.dataset_name, split_name)
            target_dataset = dataset[split_type]

            logger.success(f"Loaded {split_name} dataset successfully:")
            logger.info(f"  Samples: {len(target_dataset)}")

            # Check available columns
            has_negatives = "not_labels" in target_dataset.column_names

            if has_negatives:
                logger.info("✓ Dataset contains hard negatives (not_labels)")
            else:
                logger.info("✗ Dataset missing hard negatives")

            # Extract all unique labels
            all_labels = set()
            logger.info("Analyzing label distribution...")

            for item in tqdm(target_dataset, desc="Processing labels"):
                for label in item["labels"]:
                    all_labels.add(label)
                if has_negatives:
                    for label in item.get("not_labels", []):
                        all_labels.add(label)

            logger.info(f"Dataset statistics: {len(all_labels)} unique labels")
            return target_dataset, all_labels

        except Exception as e:
            logger.error(f"Failed to load dataset from Hugging Face: {e}")
            raise

    def prepare_contrastive_dataset(self, hf_dataset: Dataset) -> Dataset:
        """Convert HF dataset to standardized contrastive format: text, positive, negative (optional)."""
        logger.info("Preparing contrastive dataset")

        text_key = "text" if "text" in hf_dataset.column_names else "sentence"
        has_negatives = "not_labels" in hf_dataset.column_names

        dataset_dict = {"text": [], "positive": []}
        if has_negatives:
            dataset_dict["negative"] = []

        for item in tqdm(hf_dataset, desc="Creating contrastive pairs"):
            text = item[text_key]
            positive_labels = item["labels"]
            negative_labels = item.get("not_labels", []) if has_negatives else []

            # Create positive pairs
            for pos_label in positive_labels:
                dataset_dict["text"].append(text)
                dataset_dict["positive"].append(pos_label)

                if has_negatives and negative_labels:
                    # Use first negative or cycle through them
                    neg_idx = len(dataset_dict["text"]) % len(negative_labels) - 1
                    dataset_dict["negative"].append(negative_labels[neg_idx])

        logger.info(f"Created {len(dataset_dict['text'])} contrastive pairs")
        if has_negatives:
            logger.info(f"  With negatives: {len(dataset_dict['negative'])}")

        return Dataset.from_dict(dataset_dict)

    def create_triplet_evaluation_data(
        self, contrastive_dataset: Dataset
    ) -> Tuple[List[str], List[str], List[str]]:
        """Create triplet evaluation data from standardized contrastive dataset."""
        anchors = []
        positives = []
        negatives = []

        logger.info("Creating triplet evaluation data...")

        # Check if dataset has negatives
        has_negatives = "negative" in contrastive_dataset.column_names

        if not has_negatives:
            logger.warning(
                "Dataset has no negatives - using random negatives from positives"
            )
            # Collect all unique positives for random negatives
            all_positives = list(set(contrastive_dataset["positive"]))

        for i, item in enumerate(
            tqdm(contrastive_dataset, desc="Creating triplet evaluation data")
        ):
            text = item["text"]
            positive = item["positive"]

            if has_negatives:
                negative = item["negative"]
            else:
                # Use random negative from other positives
                import random

                negative = random.choice([p for p in all_positives if p != positive])

            anchors.append(text)
            positives.append(positive)
            negatives.append(negative)

        logger.info(f"Created {len(anchors)} triplets for evaluation")
        return anchors, positives, negatives

    def train(self, training_args: SentenceTransformerTrainingArguments):
        """Complete training pipeline using simplified data loading."""

        # Load training dataset
        train_hf, train_labels = self.load_dataset(self.model_args.train_split, "train")

        # Load evaluation dataset
        eval_hf, eval_labels = self.load_dataset(self.model_args.eval_split, "test")

        # Check for label overlap between train and eval
        label_overlap = train_labels & eval_labels
        if label_overlap:
            logger.warning(
                f"Found {len(label_overlap)} overlapping labels between train/eval"
            )
            logger.info(f"  Overlap examples: {list(label_overlap)[:10]}")
        else:
            logger.success("✓ No label overlap - true zero-shot setup!")

        # Prepare contrastive datasets
        train_dataset = self.prepare_contrastive_dataset(train_hf)
        eval_dataset = self.prepare_contrastive_dataset(eval_hf)

        logger.info("Datasets prepared:")
        logger.info(f"  Train: {len(train_dataset)} pairs")
        logger.info(f"  Eval: {len(eval_dataset)} pairs")

        # Initialize model
        logger.info("Initializing SentenceTransformer...")
        self.model = SentenceTransformer(
            model_name_or_path=self.model_args.model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Choose appropriate loss function - update to use 'text' instead of 'anchor'
        logger.info("Using CachedMultipleNegativesRankingLoss...")

        # Convert dataset format for compatibility with the loss function
        train_loss_dataset = Dataset.from_dict(
            {"anchor": train_dataset["text"], "positive": train_dataset["positive"]}
        )

        train_loss = losses.CachedMultipleNegativesRankingLoss(
            model=self.model,
            scale=self.model_args.scale,
            mini_batch_size=self.model_args.mini_batch_size,
            show_progress_bar=False,
            similarity_fct=util.cos_sim,
        )

        # Create TripletEvaluator using standardized evaluation data
        logger.info("Setting up TripletEvaluator...")
        anchors, positives, negatives = self.create_triplet_evaluation_data(
            eval_dataset
        )

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
            train_dataset=train_loss_dataset,
            loss=train_loss,
            evaluator=evaluator,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.model_args.early_stopping_patience
                )
            ],
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

        trainer.train()

        # Save model and metadata
        os.makedirs(training_args.output_dir, exist_ok=True)
        # self.model.save(training_args.output_dir)

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
            "has_hard_negatives_train": "negative" in train_dataset.column_names,
            "has_hard_negatives_eval": "negative" in eval_dataset.column_names,
            "total_unique_labels": len(train_labels | eval_labels),
        }

        with open(
            os.path.join(training_args.output_dir, "dataset_info.json"), "w"
        ) as f:
            json.dump(dataset_info, f, indent=2)

        # Save config
        with open(
            os.path.join(training_args.output_dir, "training_config.json"), "w"
        ) as f:
            config_dict = {**self.model_args.__dict__, **training_args.to_dict()}
            json.dump(config_dict, f, indent=2)

        # Final evaluation
        logger.info("Performing final triplet evaluation...")
        final_score = evaluator(
            self.model,
            output_path=os.path.join(
                training_args.output_dir, "final_eval_triplet.csv"
            ),
        )

        logger.success(
            f"Final triplet accuracy: {final_score[evaluator.primary_metric]:.4f}"
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
