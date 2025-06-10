import json
import os
import glob
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses, util
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers.readers import InputExample
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from loguru import logger
import random


@dataclass
class TrainingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    mini_batch_size: int = 32  # For CachedMultipleNegativesSymmetricRankingLoss
    learning_rate: float = 2e-5
    num_epochs: int = 5
    max_length: int = 512
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    negative_samples: int = 5  # Number of negative labels per positive
    scale: float = 20.0  # Scale for similarity function
    evaluation_steps: int = 1000


class ZeroShotTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.all_labels = set()
        self.model = None
        
        logger.info(f"Initialized ZeroShotTrainer with SentenceTransformer: {config.model_name}")
        
    def load_batch_data(self, batch_dir: str) -> Tuple[List[str], List[List[str]]]:
        """Load all JSON batch files and extract texts and labels."""
        texts = []
        labels = []
        label_stats = Counter()
        
        batch_files = glob.glob(os.path.join(batch_dir, "*.json"))
        batch_files.sort()
        
        logger.info(f"Found {len(batch_files)} batch files in {batch_dir}")
        
        for batch_file in tqdm(batch_files, desc="Loading batches"):
            try:
                with open(batch_file, 'r') as f:
                    batch_data = json.load(f)
                    
                for item in batch_data:
                    if 'sentence' in item and 'labels' in item:
                        texts.append(item['sentence'])
                        item_labels = item['labels']
                        labels.append(item_labels)
                        
                        # Collect all unique labels
                        for label in item_labels:
                            self.all_labels.add(label)
                            label_stats[label] += 1
                            
            except Exception as e:
                logger.warning(f"Error loading {batch_file}: {e}")
                continue
        
        logger.success(f"Loaded {len(texts)} samples with {len(self.all_labels)} unique labels")
        
        # Print label distribution
        logger.info("Top 20 most frequent labels:")
        for label, count in label_stats.most_common(20):
            logger.info(f"  {label}: {count}")
            
        return texts, labels
    
    def create_contrastive_dataset(self, texts: List[str], labels: List[List[str]]) -> Dataset:
        """
        Create dataset for contrastive learning with (anchor, positive) pairs.
        The loss function will handle negative sampling automatically using in-batch negatives.
        """
        logger.info("Creating contrastive learning dataset...")
        
        dataset_dict = {
            'anchor': [],
            'positive': []
        }
        
        total_pairs = 0
        
        for text, text_labels in tqdm(zip(texts, labels), total=len(texts), desc="Creating pairs"):
            # For each text (anchor), create pairs with all its positive labels
            for positive_label in text_labels:
                dataset_dict['anchor'].append(text)
                dataset_dict['positive'].append(positive_label)
                total_pairs += 1
        
        logger.info(f"Created {total_pairs} (anchor, positive) pairs")
        logger.info("  Anchor: text samples")
        logger.info("  Positive: labels that match the text")
        logger.info("  Negatives: handled automatically by loss function using in-batch sampling")
        
        return Dataset.from_dict(dataset_dict)
    
    def create_evaluation_data(self, texts: List[str], labels: List[List[str]]) -> Tuple[List, List]:
        """Create evaluation pairs for BinaryClassificationEvaluator."""
        eval_sentences1 = []
        eval_sentences2 = []
        eval_labels = []
        
        all_labels_list = list(self.all_labels)
        
        for text, text_labels in zip(texts, labels):
            # Positive pairs
            for label in text_labels:
                eval_sentences1.append(text)
                eval_sentences2.append(label)
                eval_labels.append(1)
            
            # Negative pairs (sample fewer for evaluation)
            negative_labels = [l for l in all_labels_list if l not in text_labels]
            if negative_labels:
                # Sample fewer negatives for evaluation
                num_negatives = min(2, len(negative_labels))
                sampled_negatives = random.sample(negative_labels, num_negatives)
                
                for neg_label in sampled_negatives:
                    eval_sentences1.append(text)
                    eval_sentences2.append(neg_label)
                    eval_labels.append(0)
        
        return list(zip(eval_sentences1, eval_sentences2)), eval_labels
    
    def train(self, prepared_data_dir: str, output_dir: str = "models/zero_shot_classifier"):
        """Complete training pipeline using pre-prepared datasets."""
        
        # Initialize wandb
        wandb.init(
            project="zero-shot-classification",
            config=self.config.__dict__,
            name=f"zero-shot-contrastive-{self.config.model_name.split('/')[-1]}"
        )
        
        # Load prepared datasets
        logger.info("Loading pre-prepared datasets...")
        train_dataset = Dataset.load_from_disk(os.path.join(prepared_data_dir, "train_contrastive"))
        test_dataset = Dataset.load_from_disk(os.path.join(prepared_data_dir, "test_contrastive"))
        
        # Load evaluation data
        with open(os.path.join(prepared_data_dir, "evaluation_data.json"), 'r') as f:
            eval_data = json.load(f)
        
        # Load label info
        with open(os.path.join(prepared_data_dir, "label_info.json"), 'r') as f:
            label_info = json.load(f)
        
        train_labels = set(label_info["train_labels"])
        test_labels = set(label_info["test_labels"])
        
        logger.info(f"Loaded datasets:")
        logger.info(f"  Train pairs: {len(train_dataset)}")
        logger.info(f"  Test pairs: {len(test_dataset)}")
        logger.info(f"  Train labels: {len(train_labels)}")
        logger.info(f"  Test labels: {len(test_labels)} (completely unseen)")
        
        # Verify no label overlap
        label_overlap = train_labels & test_labels
        if label_overlap:
            logger.warning(f"Found label overlap: {label_overlap}")
        else:
            logger.success("✓ No label overlap between train and test sets")
        
        # Initialize model
        logger.info("Initializing SentenceTransformer...")
        self.model = SentenceTransformer(
            model_name_or_path=self.config.model_name,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Create loss function - handles negatives automatically with in-batch sampling
        logger.info("Setting up CachedMultipleNegativesRankingLoss...")
        train_loss = losses.CachedMultipleNegativesRankingLoss(
            model=self.model,
            scale=self.config.scale,
            mini_batch_size=self.config.mini_batch_size,
            show_progress_bar=True,
            similarity_fct=util.cos_sim
        )
        
        # Create evaluator for UNSEEN LABELS
        logger.info("Setting up BinaryClassificationEvaluator with UNSEEN labels...")
        test_eval_pairs = eval_data["test_eval_pairs"]
        test_eval_labels = eval_data["test_eval_labels"]
        
        evaluator = BinaryClassificationEvaluator(
            sentences1=[pair[0] for pair in test_eval_pairs],
            sentences2=[pair[1] for pair in test_eval_pairs],
            labels=test_eval_labels,
            name="zero_shot_unseen_labels"
        )
        
        # Training arguments
        args = {
            'output_dir': output_dir,
            'num_train_epochs': self.config.num_epochs,
            'per_device_train_batch_size': self.config.batch_size,
            'per_device_eval_batch_size': self.config.batch_size,
            'warmup_steps': self.config.warmup_steps,
            'weight_decay': self.config.weight_decay,
            'learning_rate': self.config.learning_rate,
            'logging_steps': 100,
            'evaluation_strategy': "steps",
            'eval_steps': self.config.evaluation_steps,
            'save_steps': self.config.evaluation_steps,
            'save_strategy': "steps",
            'load_best_model_at_end': True,
            'metric_for_best_model': "zero_shot_unseen_labels_max_ap",
            'greater_is_better': True,
            'report_to': "wandb",
            'run_name': f"zero-shot-{self.config.model_name.split('/')[-1]}",
            'dataloader_num_workers': 4,
            'fp16': torch.cuda.is_available(),
        }
        
        # Initialize trainer
        logger.info("Initializing SentenceTransformerTrainer...")
        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            loss=train_loss,
            evaluator=evaluator,
        )
        
        # Train model
        logger.info("Starting contrastive training with automatic negative sampling...")
        logger.info("Training approach:")
        logger.info("  • Input: (text, positive_label) pairs")
        logger.info("  • Negatives: automatically sampled from other labels in the batch")
        logger.info("  • Goal: Learn embeddings where text and matching labels are close")
        logger.info("🎯 Key insight: Model is evaluated on completely unseen labels!")
        
        trainer.train()
        
        # Save model and metadata
        os.makedirs(output_dir, exist_ok=True)
        self.model.save(output_dir)
        
        # Copy label info to model directory
        with open(os.path.join(output_dir, 'label_info.json'), 'w') as f:
            json.dump(label_info, f, indent=2)
            
        # Save config
        with open(os.path.join(output_dir, 'training_config.json'), 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        # Final evaluation on UNSEEN labels
        logger.info("Performing final evaluation on UNSEEN labels...")
        final_score = evaluator(self.model, output_path=os.path.join(output_dir, "final_eval_unseen.csv"))
        
        logger.success(f"Final zero-shot score on UNSEEN labels: {final_score:.4f}")
        
        wandb.log({"final_eval_unseen_labels": final_score})
        wandb.finish()
        
        logger.success(f"Training completed! Model saved to {output_dir}")
        return self.model


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train zero-shot classification model using prepared datasets")
    parser.add_argument("--prepared-data-dir", type=str, required=True, help="Directory containing prepared datasets")
    parser.add_argument("--output-dir", type=str, default="models/zero_shot_classifier", help="Output directory for trained model")
    parser.add_argument("--model-name", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Base model name")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--mini-batch-size", type=int, default=32, help="Mini batch size for gradient caching")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--scale", type=float, default=20.0, help="Scale for similarity function")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        model_name=args.model_name,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        scale=args.scale
    )
    
    trainer = ZeroShotTrainer(config)
    trainer.train(args.prepared_data_dir, args.output_dir)


if __name__ == "__main__":
    main()
