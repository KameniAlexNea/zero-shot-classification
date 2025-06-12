#!/usr/bin/env python3
"""
Training script for FZeroNet - Zero-shot Classification Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
import argparse
import logging
from pathlib import Path
import time

from gliznet.tokenizer import ZeroShotClassificationTokenizer
from gliznet.model import FZeroNet

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ZeroShotDataset(Dataset):
    """
    Dataset class for zero-shot classification training.
    """
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: ZeroShotClassificationTokenizer,
        max_labels: Optional[int] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data: List of dictionaries with 'text', 'positive_labels', 'negative_labels'
            tokenizer: Tokenizer instance
            max_labels: Maximum number of labels to consider (for padding consistency)
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_labels = max_labels
        
        # Determine max_labels if not provided
        if self.max_labels is None:
            self.max_labels = max(
                len(item['positive_labels']) + len(item['negative_labels'])
                for item in data
            )
        
        logger.info(f"Dataset initialized with {len(data)} samples, max_labels: {self.max_labels}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict:
        item = self.data[idx]
        
        text = item['text']
        positive_labels = item['positive_labels']
        negative_labels = item['negative_labels']
        all_labels = positive_labels + negative_labels
        
        # Pad or truncate labels to max_labels
        if len(all_labels) > self.max_labels:
            all_labels = all_labels[:self.max_labels]
            positive_labels = positive_labels[:min(len(positive_labels), self.max_labels)]
            negative_labels = negative_labels[:max(0, self.max_labels - len(positive_labels))]
        elif len(all_labels) < self.max_labels:
            # Pad with dummy labels
            padding_needed = self.max_labels - len(all_labels)
            all_labels.extend([f"dummy_label_{i}" for i in range(padding_needed)])
        
        # Create ground truth
        ground_truth = [0.0] * self.max_labels
        for i in range(len(positive_labels)):
            ground_truth[i] = 1.0
        
        # Tokenize
        tokenized = self.tokenizer.tokenize_example(
            text, all_labels, return_tensors="pt"
        )
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'label_mask': tokenized['label_mask'].squeeze(0),
            'labels': torch.tensor(ground_truth, dtype=torch.float32),
            'text': text,
            'positive_labels': positive_labels,
            'negative_labels': negative_labels,
            'all_labels': all_labels
        }


def create_sample_dataset() -> List[Dict]:
    """Create a sample dataset for demonstration."""
    return [
        {
            "text": "Scientists at Stanford University have developed a new artificial intelligence system that can predict protein structures with unprecedented accuracy.",
            "positive_labels": ["artificial_intelligence", "scientific_research", "university_news"],
            "negative_labels": ["sports_news", "cooking_recipe", "travel_guide"]
        },
        {
            "text": "The local basketball team won their championship game last night with a final score of 98-87.",
            "positive_labels": ["sports_news", "basketball", "local_news"],
            "negative_labels": ["scientific_research", "cooking_recipe", "technology_news"]
        },
        {
            "text": "This delicious pasta recipe combines traditional Italian techniques with modern molecular gastronomy.",
            "positive_labels": ["cooking_recipe", "food_preparation", "culinary_arts"],
            "negative_labels": ["sports_news", "scientific_research", "technology_news"]
        },
        {
            "text": "Apple announced their latest iPhone model featuring advanced camera capabilities and improved battery life.",
            "positive_labels": ["technology_news", "product_announcement", "mobile_devices"],
            "negative_labels": ["cooking_recipe", "sports_news", "medical_advice"]
        },
        {
            "text": "Researchers have discovered that regular exercise can significantly reduce the risk of cardiovascular disease.",
            "positive_labels": ["medical_research", "health_advice", "scientific_study"],
            "negative_labels": ["technology_news", "cooking_recipe", "sports_news"]
        },
        {
            "text": "The complete guide to visiting Paris includes recommendations for museums, restaurants, and hidden gems.",
            "positive_labels": ["travel_guide", "tourism", "cultural_information"],
            "negative_labels": ["scientific_research", "sports_news", "technology_news"]
        }
    ]


def train_model(
    model: FZeroNet,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    save_path: Optional[str] = None
) -> Dict:
    """
    Train the FZeroNet model.
    
    Args:
        model: FZeroNet model instance
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        optimizer: Optimizer instance
        num_epochs: Number of training epochs
        device: Device to train on
        save_path: Path to save the model (optional)
        
    Returns:
        Training history dictionary
    """
    model.to(device)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    logger.info(f"Starting training for {num_epochs} epochs on {device}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label_mask = batch['label_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                label_mask=label_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            epoch_val_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    # Move batch to device
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    label_mask = batch['label_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    # Forward pass
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        label_mask=label_mask,
                        labels=labels
                    )
                    
                    loss = outputs['loss']
                    epoch_val_loss += loss.item()
                    val_batches += 1
                    
                    # Calculate accuracy
                    logits = outputs['logits']
                    predictions = (torch.sigmoid(logits) > 0.5).float()
                    correct_predictions += (predictions == labels).sum().item()
                    total_predictions += labels.numel()
            
            avg_val_loss = epoch_val_loss / val_batches
            val_accuracy = correct_predictions / total_predictions
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        else:
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")
    
    # Save model if path provided
    if save_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
        }, save_path)
        logger.info(f"Model saved to {save_path}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }


def evaluate_model(
    model: FZeroNet,
    test_loader: DataLoader,
    device: torch.device
) -> Dict:
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained FZeroNet model
        test_loader: Test data loader
        device: Device to run evaluation on
        
    Returns:
        Evaluation metrics
    """
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    num_batches = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label_mask = batch['label_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                label_mask=label_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            total_loss += loss.item()
            num_batches += 1
            
            # Get predictions
            logits = outputs['logits']
            predictions = (torch.sigmoid(logits) > 0.5).float()
            
            # Calculate accuracy
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.numel()
            
            # Store for detailed analysis
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / num_batches
    accuracy = correct_predictions / total_predictions
    
    # Calculate precision, recall, F1 (simplified for multi-label)
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    true_positives = np.sum((all_predictions == 1) & (all_labels == 1))
    false_positives = np.sum((all_predictions == 1) & (all_labels == 0))
    false_negatives = np.sum((all_predictions == 0) & (all_labels == 1))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def main():
    parser = argparse.ArgumentParser(description='Train FZeroNet for zero-shot classification')
    parser.add_argument('--model_name', default='bert-base-uncased', help='HuggingFace model name')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size for projections')
    parser.add_argument('--similarity_metric', default='cosine', choices=['cosine', 'dot', 'bilinear'])
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--device', default='auto', help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--save_path', default='fzeronet_model.pt', help='Path to save the model')
    parser.add_argument('--data_path', help='Path to training data JSON file (optional)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load or create dataset
    if args.data_path and Path(args.data_path).exists():
        logger.info(f"Loading data from {args.data_path}")
        with open(args.data_path, 'r') as f:
            data = json.load(f)
    else:
        logger.info("Using sample dataset")
        data = create_sample_dataset()
    
    # Split data (80% train, 20% validation)
    split_idx = int(0.8 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:] if len(data) > 1 else None
    
    logger.info(f"Training samples: {len(train_data)}")
    if val_data:
        logger.info(f"Validation samples: {len(val_data)}")
    
    # Initialize tokenizer and model
    tokenizer = ZeroShotClassificationTokenizer(model_name=args.model_name)
    model = FZeroNet(
        model_name=args.model_name,
        hidden_size=args.hidden_size,
        similarity_metric=args.similarity_metric
    )
    
    # Create datasets and data loaders
    train_dataset = ZeroShotDataset(train_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    val_loader = None
    if val_data:
        val_dataset = ZeroShotDataset(val_data, tokenizer, max_labels=train_dataset.max_labels)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        device=device,
        save_path=args.save_path
    )
    
    # Evaluate on validation set
    if val_loader:
        logger.info("Evaluating on validation set...")
        eval_results = evaluate_model(model, val_loader, device)
        logger.info(f"Validation Results:")
        logger.info(f"  Loss: {eval_results['loss']:.4f}")
        logger.info(f"  Accuracy: {eval_results['accuracy']:.4f}")
        logger.info(f"  Precision: {eval_results['precision']:.4f}")
        logger.info(f"  Recall: {eval_results['recall']:.4f}")
        logger.info(f"  F1 Score: {eval_results['f1_score']:.4f}")
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
