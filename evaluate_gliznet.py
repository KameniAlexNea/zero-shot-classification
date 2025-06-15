import json
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import load_file
from tqdm import tqdm

from gliznet.model import GliZNetModel
from gliznet.tokenizer import GliZNETTokenizer, load_dataset


def load_models(
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    logger.info(f"Loading model from {model_path} on {device}")

    # Load checkpoint
    state_dict = load_file(model_path, device=str(device))

    # Initialize tokenizer
    tokenizer = GliZNETTokenizer(model_name=model_name)

    # Initialize model
    model = GliZNetModel(model_name=model_name, hidden_size=256)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info("Model loaded successfully")
    return model, tokenizer


data = load_dataset(split="test")
model, tokenizer = load_models(
    model_path="results/checkpoint-1614/model.safetensors",
    model_name="bert-base-uncased",
    device="auto",
)

max_labels = 20
all_predictions = []
for batch in tqdm(data.iter(batch_size=64)):
    sentences = batch["text"]
    labels = batch["labels_text"]
    masks = batch["labels_int"]
    inputs_texts = []
    labels_texts = []
    labels_logits = []

    for sentence, label, mask in zip(sentences, labels, masks):
        for i in range(0, len(label), max_labels):
            inputs_texts.append(sentence)
            labels_texts.append(label[i : i + max_labels])
            labels_logits.append(mask[i : i + max_labels])

    inputs = tokenizer(
        inputs_texts,
        labels=labels_texts,
        pad=True,
        return_tensors="pt",
    )
    model_predictions = model.predict(**{k: v.to("cuda") for k, v in inputs.items()})

    all_predictions.append(
        {
            "sentences": inputs_texts,
            "labels": labels_texts,
            "masks": labels_logits,
            "predictions": model_predictions,
        }
    )

# Process predictions to calculate metrics
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0
all_results = []

for batch_predictions in all_predictions:
    for sentence, labels, masks, preds in zip(
        batch_predictions["sentences"],
        batch_predictions["labels"],
        batch_predictions["masks"],
        batch_predictions["predictions"],
    ):
        # Convert predictions to binary values (assuming model outputs probabilities)
        binary_preds = [1 if p >= 0.5 else 0 for p in preds]

        # Calculate TP, FP, TN, FN for each prediction
        for j in range(len(binary_preds)):
            if j < len(masks):  # Ensure we're within bounds
                if masks[j] == 1 and binary_preds[j] == 1:
                    true_positives += 1
                elif masks[j] == 0 and binary_preds[j] == 1:
                    false_positives += 1
                elif masks[j] == 0 and binary_preds[j] == 0:
                    true_negatives += 1
                elif masks[j] == 1 and binary_preds[j] == 0:
                    false_negatives += 1

        # Store individual result
        all_results.append(
            {
                "sentence": sentence,
                "true_labels": [
                    {"label": label, "is_positive": mask == 1}
                    for label, mask in zip(labels, masks)
                ],
                "predictions": [
                    {"label": label, "score": float(pred), "predicted": pred >= 0.5}
                    for label, pred in zip(labels, preds)
                ],
            }
        )

# Calculate metrics
precision = (
    true_positives / (true_positives + false_positives)
    if (true_positives + false_positives) > 0
    else 0
)
recall = (
    true_positives / (true_positives + false_negatives)
    if (true_positives + false_negatives) > 0
    else 0
)
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
accuracy = (true_positives + true_negatives) / (
    true_positives + true_negatives + false_positives + false_negatives
)

# Print metrics
logger.info("Evaluation metrics:")
logger.info(f"Precision: {precision:.4f}")
logger.info(f"Recall: {recall:.4f}")
logger.info(f"F1 Score: {f1:.4f}")
logger.info(f"Accuracy: {accuracy:.4f}")

# Save results to JSON
results_dir = Path("results/evaluation")
results_dir.mkdir(parents=True, exist_ok=True)
results_path = results_dir / "predictions.json"

with open(results_path, "w") as f:
    json.dump(
        {
            "metrics": {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
            },
            "predictions": all_results,
        },
        f,
        indent=2,
    )

logger.info(f"Results saved to {results_path}")

with open(results_dir / "all_predictions.json", "w") as f:
    json.dump(all_predictions, f, indent=2, ensure_ascii=False)
