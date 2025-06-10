import json
import os
import glob
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    TrainingArguments, Trainer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from loguru import logger
import pickle
import random


@dataclass
class TrainingConfig:
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    label_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # Can be different
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 5
    max_length: int = 512
    warmup_steps: int = 500
    weight_decay: float = 0.01
    dropout: float = 0.1
    temperature: float = 0.1
    negative_samples: int = 3  # Number of negative labels per positive


class ZeroShotCrossEncoder(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Use a classification model as base
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.text_model_name, 
            num_labels=2,  # Binary: match (1) or no match (0)
            hidden_dropout_prob=config.dropout,
            attention_probs_dropout_prob=config.dropout
        )
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs


class ZeroShotTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
        self.all_labels = set()
        self.model = None
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info(f"Initialized ZeroShotTrainer with model: {config.text_model_name}")
        
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
    
    def create_training_pairs(self, texts: List[str], labels: List[List[str]]) -> Dict[str, List]:
        """
        Create positive and negative training pairs for zero-shot classification.
        
        Returns:
            Dictionary with 'text', 'label', 'is_match' keys for Dataset creation
        """
        logger.info("Creating positive and negative training pairs...")
        
        training_data = {
            'text': [],
            'label': [],
            'is_match': []  # 1 for positive pairs, 0 for negative pairs
        }
        
        all_labels_list = list(self.all_labels)
        positive_count = 0
        negative_count = 0
        
        for text, text_labels in tqdm(zip(texts, labels), total=len(texts), desc="Creating pairs"):
            # POSITIVE PAIRS: text matches its assigned labels
            for label in text_labels:
                training_data['text'].append(text)
                training_data['label'].append(label)
                training_data['is_match'].append(1)  # Positive class
                positive_count += 1
            
            # NEGATIVE PAIRS: text does NOT match random other labels
            negative_labels = [l for l in all_labels_list if l not in text_labels]
            
            if negative_labels:
                # Sample negative labels
                num_negatives = min(self.config.negative_samples, len(negative_labels))
                sampled_negatives = random.sample(negative_labels, num_negatives)
                
                for neg_label in sampled_negatives:
                    training_data['text'].append(text)
                    training_data['label'].append(neg_label)
                    training_data['is_match'].append(0)  # Negative class
                    negative_count += 1
        
        logger.info(f"Created training pairs:")
        logger.info(f"  Positive pairs (text matches label): {positive_count}")
        logger.info(f"  Negative pairs (text does NOT match label): {negative_count}")
        logger.info(f"  Total pairs: {len(training_data['text'])}")
        logger.info(f"  Class balance: {positive_count / (positive_count + negative_count):.2%} positive")
        
        return training_data
    
    def prepare_datasets(self, texts: List[str], labels: List[List[str]], 
                        train_ratio: float = 0.8) -> Tuple[Dataset, Dataset]:
        """Prepare train and validation datasets using datasets library."""
        
        # Split data
        split_idx = int(len(texts) * train_ratio)
        train_texts, val_texts = texts[:split_idx], texts[split_idx:]
        train_labels, val_labels = labels[:split_idx], labels[split_idx:]
        
        logger.info(f"Training samples: {len(train_texts)}")
        logger.info(f"Validation samples: {len(val_texts)}")
        logger.info(f"Total unique labels: {len(self.all_labels)}")
        
        # Create training pairs
        train_data = self.create_training_pairs(train_texts, train_labels)
        val_data = self.create_training_pairs(val_texts, val_labels)
        
        # Create datasets using datasets library
        train_dataset = Dataset.from_dict(train_data)
        val_dataset = Dataset.from_dict(val_data)
        
        # Tokenize datasets
        def tokenize_function(examples):
            # Cross-encoder: concatenate text and label with [SEP]
            return self.tokenizer(
                examples['text'],
                examples['label'],
                truncation=True,
                padding=False,  # Will be handled by data collator
                max_length=self.config.max_length,
            )
        
        # Apply tokenization
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text', 'label']  # Keep only tokenized inputs and labels
        )
        
        val_dataset = val_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text', 'label']
        )
        
        # Rename 'is_match' to 'labels' for compatibility with Trainer
        train_dataset = train_dataset.rename_column('is_match', 'labels')
        val_dataset = val_dataset.rename_column('is_match', 'labels')
        
        logger.info(f"Training pairs after tokenization: {len(train_dataset)}")
        logger.info(f"Validation pairs after tokenization: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        
        # Calculate class-specific metrics
        pos_f1 = f1_score(labels, predictions, pos_label=1)
        neg_f1 = f1_score(labels, predictions, pos_label=0)
        
        return {
            'accuracy': accuracy,
            'f1_weighted': f1,
            'f1_positive': pos_f1,
            'f1_negative': neg_f1
        }
    
    def train(self, batch_dir: str, output_dir: str = "models/zero_shot_classifier"):
        """Complete training pipeline using Transformers Trainer."""
        
        # Initialize wandb
        wandb.init(
            project="zero-shot-classification",
            config=self.config.__dict__,
            name=f"zero-shot-cross-encoder-{self.config.text_model_name.split('/')[-1]}"
        )
        
        # Load data
        logger.info("Loading training data...")
        texts, labels = self.load_batch_data(batch_dir)
        
        # Prepare datasets
        logger.info("Preparing datasets...")
        train_dataset, val_dataset = self.prepare_datasets(texts, labels)
        
        # Initialize model
        self.model = ZeroShotCrossEncoder(self.config)
        
        # Resize token embeddings if needed
        self.model.model.resize_token_embeddings(len(self.tokenizer))
        
        # Data collator for dynamic padding
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt"
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            eval_steps=500,
            save_steps=1000,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="f1_weighted",
            greater_is_better=True,
            report_to="wandb",
            run_name=f"zero-shot-{self.config.text_model_name.split('/')[-1]}",
            dataloader_num_workers=4,
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            remove_unused_columns=False,  # Keep all columns
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train model
        logger.info("Starting training...")
        logger.info("Model will learn to classify:")
        logger.info("  Class 0 (Negative): Text does NOT match the given label")
        logger.info("  Class 1 (Positive): Text MATCHES the given label")
        
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save label vocabulary
        with open(os.path.join(output_dir, 'all_labels.json'), 'w') as f:
            json.dump(list(self.all_labels), f, indent=2)
            
        # Save config
        with open(os.path.join(output_dir, 'training_config.json'), 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        # Final evaluation
        logger.info("Performing final evaluation...")
        eval_results = trainer.evaluate()
        
        logger.info("Final Evaluation Results:")
        for key, value in eval_results.items():
            logger.info(f"  {key}: {value:.4f}")
        
        # Save evaluation results
        with open(os.path.join(output_dir, 'eval_results.json'), 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        self._plot_training_metrics(trainer, output_dir)
        wandb.finish()
        
        logger.success(f"Training completed! Model saved to {output_dir}")
        logger.success("Model is now ready for zero-shot classification!")

    def _plot_training_metrics(self, trainer, output_dir: str):
        """Plot training metrics."""
        if hasattr(trainer.state, 'log_history'):
            train_losses = []
            eval_losses = []
            eval_f1s = []
            
            for log in trainer.state.log_history:
                if 'train_loss' in log:
                    train_losses.append(log['train_loss'])
                if 'eval_loss' in log:
                    eval_losses.append(log['eval_loss'])
                if 'eval_f1' in log:
                    eval_f1s.append(log['eval_f1'])
            
            # Plot losses
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            if train_losses:
                plt.plot(train_losses, label='Train Loss')
            if eval_losses:
                plt.plot(eval_losses, label='Eval Loss')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            if eval_f1s:
                plt.plot(eval_f1s, label='Eval F1', color='green')
            plt.xlabel('Steps')
            plt.ylabel('F1 Score')
            plt.title('Validation F1 Score')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Training metrics plot saved!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train zero-shot classification model")
    parser.add_argument("--batch-dir", type=str, required=True, help="Directory containing batch JSON files")
    parser.add_argument("--output-dir", type=str, default="models/zero_shot_classifier", help="Output directory for trained model")
    parser.add_argument("--text-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Text encoder model name")
    parser.add_argument("--label-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Label encoder model name")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--negative-samples", type=int, default=3, help="Number of negative samples per positive")
    
    args = parser.parse_args()
    
    # Create training configuration
    config = TrainingConfig(
        text_model_name=args.text_model,
        label_model_name=args.label_model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        negative_samples=args.negative_samples
    )
    
    # Initialize and run trainer
    trainer = ZeroShotTrainer(config)
    trainer.train(args.batch_dir, args.output_dir)


if __name__ == "__main__":
    main()
