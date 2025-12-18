"""Usage examples for GliZNet models.

This module demonstrates best practices for training and using GliZNet models
with the improved configuration system.
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers import BertConfig

from .config import (
    GliZNetDataConfig,
    GliZNetTrainingConfig,
    get_separator_pooling_config,
)
from .data import add_tokenized_function, load_dataset
from .model import GliZNetForSequenceClassification
from .tokenizer import GliZNETTokenizer


def create_model_with_config(
    pretrained_model_name: str = "bert-base-uncased",
    use_custom_separator: bool = True,
    training_config: Optional[GliZNetTrainingConfig] = None,
):
    """Create a GliZNet model with proper configuration.

    Args:
        pretrained_model_name: HuggingFace model identifier
        use_custom_separator: If True, use [LAB] token; if False, use semicolon
        training_config: Custom training configuration (uses default if None)

    Returns:
        Tuple of (model, tokenizer)

    Example:
        >>> model, tokenizer = create_model_with_config()
        >>> # Model is ready for training
    """
    # Create tokenizer with appropriate separator
    separator = "[LAB]" if use_custom_separator else ";"
    tokenizer = GliZNETTokenizer.from_pretrained(
        pretrained_model_name,
        cls_separator_token=separator,
    )

    # Use provided config or create default based on pooling strategy
    if training_config is None:
        if use_custom_separator:
            training_config = get_separator_pooling_config()
        else:
            from .config import get_mean_pooling_config

            training_config = get_mean_pooling_config()

    # Create model configuration
    bert_config = BertConfig.from_pretrained(pretrained_model_name)

    # Initialize model with training config
    model = GliZNetForSequenceClassification(
        config=bert_config, **training_config.to_model_kwargs()
    )

    # Resize embeddings if using custom tokens
    if tokenizer.has_custom_tokens():
        new_vocab_size = tokenizer.get_vocab_size()
        model.resize_token_embeddings(new_vocab_size)
        model.config.vocab_size = new_vocab_size
        model.config.use_separator_pooling = True

    print(f"✓ Model created with pooling strategy: {tokenizer.get_pooling_strategy()}")
    print(f"✓ Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"✓ Loss configuration:")
    print(f"  - BCE scale: {training_config.scale_loss}")
    print(f"  - Contrastive weight: {training_config.contrastive_loss_weight}")
    print(f"  - Barlow weight: {training_config.barlow_loss_weight}")

    return model, tokenizer


def prepare_dataset(
    dataset_path: str,
    tokenizer: GliZNETTokenizer,
    data_config: Optional[GliZNetDataConfig] = None,
    split: str = "train",
):
    """Prepare a dataset for training.

    Args:
        dataset_path: HuggingFace dataset path
        tokenizer: GliZNETTokenizer instance
        data_config: Data processing configuration
        split: Dataset split to load

    Returns:
        Tokenized dataset ready for DataLoader

    Example:
        >>> tokenizer = GliZNETTokenizer.from_pretrained("bert-base-uncased")
        >>> dataset = prepare_dataset("username/dataset", tokenizer)
        >>> dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
    """
    if data_config is None:
        data_config = GliZNetDataConfig()

    # Load raw dataset
    raw_dataset = load_dataset(
        path=dataset_path,
        split=split,
        text_column=data_config.text_column,
        positive_column=data_config.positive_column,
        negative_column=data_config.negative_column,
        shuffle_labels=data_config.shuffle_labels,
        min_label_length=data_config.min_label_length,
    )

    # Tokenize dataset
    tokenized_dataset = add_tokenized_function(
        raw_dataset,
        tokenizer=tokenizer,
        max_labels=data_config.max_labels,
        shuffle_labels=data_config.shuffle_labels,
        as_transform=True,
    )

    print(f"✓ Dataset prepared: {len(tokenized_dataset)} samples")
    print(f"✓ Max labels per sample: {data_config.max_labels}")
    print(f"✓ Shuffle labels: {data_config.shuffle_labels}")

    return tokenized_dataset


def example_training_setup():
    """Complete example of setting up model and data for training."""
    print("=" * 60)
    print("GliZNet Training Setup Example")
    print("=" * 60)

    # 1. Create custom training configuration
    training_config = GliZNetTrainingConfig(
        projected_dim=256,  # Project to 256 dimensions
        similarity_metric="dot",
        scale_loss=10.0,
        barlow_loss_weight=0.05,
        contrastive_loss_weight=1.0,
        use_separator_pooling=True,  # Use separator token pooling
    )

    print("\n1. Training Configuration:")
    print(f"   Projected dim: {training_config.projected_dim}")
    print(f"   Similarity: {training_config.similarity_metric}")
    print(f"   Use separator pooling: {training_config.use_separator_pooling}")

    # 2. Create model and tokenizer
    print("\n2. Creating Model and Tokenizer:")
    model, tokenizer = create_model_with_config(
        pretrained_model_name="bert-base-uncased",
        use_custom_separator=True,
        training_config=training_config,
    )

    # 3. Prepare data configuration
    data_config = GliZNetDataConfig(
        max_labels=30,  # Limit to 30 labels per sample
        shuffle_labels=True,
        min_label_length=2,
    )

    print("\n3. Data Configuration:")
    print(f"   Max labels: {data_config.max_labels}")
    print(f"   Shuffle: {data_config.shuffle_labels}")

    # 4. Example prediction
    print("\n4. Example Usage:")
    example_text = "The researchers discovered ancient tools."
    example_labels = ["archaeology", "science", "cooking", "sports"]

    # Tokenize
    inputs = tokenizer(example_text, example_labels)

    # Add batch dimension
    for key in inputs:
        inputs[key] = inputs[key].unsqueeze(0)

    # Predict (example - model not trained)
    model.eval()
    with torch.no_grad():
        predictions = model.predict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            lmask=inputs["lmask"],
        )

    print(f"   Text: {example_text}")
    print(f"   Labels: {example_labels}")
    print(f"   Predictions: {predictions[0]}")

    print("\n" + "=" * 60)
    print("Setup complete! Ready for training.")
    print("=" * 60)

    return model, tokenizer, data_config


def example_inference():
    """Example of using a trained model for inference."""
    print("\n" + "=" * 60)
    print("GliZNet Inference Example")
    print("=" * 60)

    # Load model and tokenizer (assuming already trained)
    model, tokenizer = create_model_with_config(use_custom_separator=True)
    model.eval()

    # Example texts and candidate labels
    texts = [
        "SpaceX launches new satellite into orbit",
        "New restaurant opens downtown",
    ]

    labels_batch = [
        ["space", "technology", "food", "sports"],
        ["food", "business", "technology", "entertainment"],
    ]

    print("\nPerforming batch inference...")

    # Tokenize batch
    batch_inputs = tokenizer(texts, labels_batch)

    # Predict
    with torch.no_grad():
        predictions = model.predict(
            input_ids=batch_inputs["input_ids"],
            attention_mask=batch_inputs["attention_mask"],
            lmask=batch_inputs["lmask"],
        )

    # Display results
    for text, labels, scores in zip(texts, labels_batch, predictions):
        print(f"\nText: {text}")
        print("Predictions:")
        for label, score in zip(labels, scores):
            print(f"  {label:20s} {score:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Run example
    example_training_setup()
    example_inference()
