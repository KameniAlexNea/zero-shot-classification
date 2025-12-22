"""Usage examples for GliZNet models.

This module demonstrates best practices for training and using GliZNet models
with the improved configuration system.
"""

from __future__ import annotations

from typing import Optional

import torch

from gliznet.config import GliZNetDataConfig
from gliznet.data import add_tokenized_function, load_dataset
from gliznet.model import GliZNetConfig, GliZNetForSequenceClassification
from gliznet.predictor import GliZNetPredictor
from gliznet.tokenizer import GliZNETTokenizer


def create_model_with_config(
    pretrained_model_name: str = "answerdotai/ModernBERT-base",
):
    """Create a GliZNet model with proper configuration.

    Args:
        pretrained_model_name: HuggingFace model identifier
        training_config: Custom training configuration (uses default if None)

    Returns:
        Tuple of (model, tokenizer)

    Example:
        >>> model, tokenizer = create_model_with_config()
        >>> # Model is ready for training
    """
    # Create tokenizer with appropriate separator
    separator = "[LAB]"
    tokenizer = GliZNETTokenizer.from_pretrained(
        pretrained_model_name, lab_cls_token=separator, fix_mistral_regex=True
    )

    model_config = GliZNetConfig(backbone_model=pretrained_model_name)
    # Initialize model with training config
    model = GliZNetForSequenceClassification.from_backbone_pretrained(
        model_config, tokenizer
    )

    print(f"✓ Model created with {pretrained_model_name} backbone.")
    print(f"✓ Vocabulary size: {tokenizer.vocab_size}")
    print(f"Similarity metric: {model_config.similarity_metric}")
    print("Model Aggregator Shape")
    print(model.aggregator)
    print("Similarity head layer")
    print(model.aggregator.similarity_head)

    return model, tokenizer


def prepare_dataset(
    dataset_path: str,
    tokenizer: GliZNETTokenizer,
    data_config: Optional[GliZNetDataConfig] = None,
    split: str = "test",
):
    """Prepare a dataset for training.

    Args:
        dataset_path: HuggingFace dataset path
        tokenizer: GliZNETTokenizer instance
        data_config: Data processing configuration
        split: Dataset split to load

    Returns:
        Tokenized dataset ready for DataLoader
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

    # 2. Create model and tokenizer
    print("\n2. Creating Model and Tokenizer:")
    model, tokenizer = create_model_with_config(
        pretrained_model_name="answerdotai/ModernBERT-base",
    )

    # 2a. Tokenizer structure visualization
    print("\n2a. Tokenizer Structure Visualization:")
    example_text = "The researchers discovered ancient tools."
    example_labels = ["archaeology", "science", "cooking"]

    # Single tokenization
    single_inputs: dict[str, torch.Tensor] = tokenizer([(example_text, example_labels)])

    # Decode to show structure
    tokens_decoded = []

    for token_id, lmask_val in zip(
        single_inputs["input_ids"][0].tolist(), single_inputs["lmask"][0].tolist()
    ):
        if token_id == tokenizer.pad_token_id:
            break
        token_str = tokenizer.decode([token_id])
        tokens_decoded.append(f"{token_str}(lmask={lmask_val})")

    print(f"   Text: {example_text}")
    print(f"   Labels: {example_labels}")
    print(f"   Token structure: {' '.join(tokens_decoded)}")
    print(
        f"   Input IDs (non-pad): {single_inputs['input_ids'][single_inputs['input_ids'] != tokenizer.pad_token_id].tolist()}"
    )
    print(
        f"   Lmask (non-pad): {single_inputs['lmask'][single_inputs['lmask'] != 0].tolist()}"
    )

    # 2b. Compare single vs batch tokenization
    print("\n2b. Single vs Batch Tokenization Comparison:")
    texts = [example_text, "Another test sentence here."]
    labels_batch = [example_labels, ["testing", "comparison"]]

    # Single processing
    single_results = []
    for text, labels in zip(texts, labels_batch):
        result = tokenizer.tokenize(text, labels)
        single_results.append(result)

    # Batch processing
    batch_results = tokenizer.tokenize(texts, labels_batch)
    # Compare
    print("   Comparing single vs batch tokenization:")
    for idx in range(len(texts)):
        single_ids = single_results[idx]["input_ids"]
        batch_ids = batch_results["input_ids"][idx]
        single_lmask = single_results[idx]["lmask"]
        batch_lmask = batch_results["lmask"][idx]

        ids_match = torch.equal(single_ids, batch_ids)
        lmask_match = torch.equal(single_lmask, batch_lmask)

        print(
            f"   Sample {idx}: input_ids match={ids_match}, lmask match={lmask_match}"
        )
        if not ids_match:
            print(
                f"      Single IDs: {single_ids[single_ids != tokenizer.pad_token_id].tolist()}"
            )
            print(
                f"      Batch IDs:  {batch_ids[batch_ids != tokenizer.pad_token_id].tolist()}"
            )
        if not lmask_match:
            print(f"      Single lmask: {single_lmask[single_lmask != 0].tolist()}")
            print(f"      Batch lmask:  {batch_lmask[batch_lmask != 0].tolist()}")

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
    inputs = tokenizer.tokenize(example_text, example_labels)

    # Add batch dimension
    for key in inputs:
        inputs[key] = inputs[key].unsqueeze(0)

    # Predict (example - model not trained)
    predictor = GliZNetPredictor(model=model, tokenizer=tokenizer)
    with torch.no_grad():
        predictions = predictor.predict(
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
    model, tokenizer = create_model_with_config()
    model.eval()
    predictor = GliZNetPredictor(model=model, tokenizer=tokenizer)

    # Example texts and candidate labels
    texts = [
        "SpaceX launches new satellite into orbit",
        "New restaurant opens downtown",
        "The World Health Organization has released a comprehensive report detailing the latest advances in medical research and public health initiatives across multiple continents. The report highlights significant progress in vaccine development, disease prevention strategies, and healthcare infrastructure improvements in developing nations. Researchers have identified several key factors contributing to improved health outcomes, including increased access to clean water, better nutrition programs, and enhanced medical training for healthcare workers in remote areas. The document also addresses ongoing challenges such as antimicrobial resistance, emerging infectious diseases, and the need for sustainable funding mechanisms to support long-term health programs. International collaboration between governments, non-governmental organizations, and private sector partners has been instrumental in achieving these milestones, demonstrating the power of coordinated global efforts in addressing complex health challenges.",
    ]

    labels_batch = [
        ["space", "technology", "food", "sports"],
        ["food", "business", "technology", "entertainment"],
        ["food", "business", "technology", "entertainment"],
    ]

    print("\nPerforming batch inference...")

    outputs = predictor.predict_batch(
        texts=texts,
        all_labels=labels_batch,
        tokenizer=tokenizer,
        activation="sigmoid",
    )

    # Display results
    for prediction in outputs:
        print(f"\nText: {prediction.text}")
        print("Predictions:")
        for labs in prediction.labels:
            print(f"  {labs.label:20s} {labs.score:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Run example
    example_training_setup()
    example_inference()
