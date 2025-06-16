#!/usr/bin/env python3

import os
os.environ["WANDB_PROJECT"] = "gliznet"
os.environ["WANDB_WATCH"] = "none"

import torch
from loguru import logger
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, HfArgumentParser

from gliznet.data import GliZNetDataset, collate_fn
from gliznet.model import GliZNetModel
from gliznet.tokenizer import GliZNETTokenizer, load_dataset

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from sklearn.metrics import roc_auc_score

def compute_metrics(eval_pred):
    """
    Compute classification metrics on the flattened, unmasked positions.
    """
    logits, label_ids = eval_pred.predictions, eval_pred.label_ids
    # if predictions/labels come as list of arrays with variable lengths, concatenate
    if isinstance(logits, list):
        logits = np.concatenate([l.flatten() for l in logits], axis=0)
        label_ids = np.concatenate([l.flatten() for l in label_ids], axis=0)
    # sigmoid for scores and threshold at 0.5
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    # now flatten (they are 1D already after concat)
    y_true = label_ids
    y_pred = preds
    y_scores = probs
    # ignore padding / unlabeled positions marked as -100
    mask = y_true != -100
    y_true, y_pred, y_scores = y_true[mask], y_pred[mask], y_scores[mask]

    # confusion
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    # core metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    balanced_accuracy = (recall + specificity) / 2

    # AUC-ROC
    try:
        auc_roc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0
    except Exception:
        auc_roc = 0.0

    # MCC
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = float((tp * tn - fp * fn) / denom) if denom > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "auc_roc": auc_roc,
        "mcc": mcc,
    }

def main():
    @dataclass
    class ModelArgs:
        model_name: str = field(default="bert-base-uncased", metadata={"help": "Pretrained model name or path"})
        hidden_size: int = field(default=256, metadata={"help": "Hidden size for projection layer"})
        similarity_metric: str = field(default="bilinear", metadata={"help": "Similarity metric"})
        max_labels: Optional[int] = field(default=None, metadata={"help": "Maximum number of labels"})
        shuffle_labels: bool = field(default=True, metadata={"help": "Shuffle labels"})
        save_path: str = field(default="models/fzeronet_model.pt", metadata={"help": "Legacy model save path"})

    parser = HfArgumentParser((ModelArgs, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    # Set device
    device = "cuda" if torch.cuda.is_available() and not training_args.no_cuda else "cpu"
    logger.info(f"Using device: {device}")

    # Load and prepare dataset
    dataset = load_dataset(
        max_labels=model_args.max_labels,
        shuffle_labels=model_args.shuffle_labels,
    )
    splits = dataset.train_test_split(test_size=0.1, seed=42)
    train_data = splits["train"]
    val_data = splits["test"]

    logger.info(
        f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}"
    )

    # Initialize tokenizer
    tokenizer = GliZNETTokenizer.from_pretrained(model_args.model_name)

    # Create datasets
    train_dataset = GliZNetDataset(hf_dataset=train_data, tokenizer=tokenizer)

    val_dataset = GliZNetDataset(hf_dataset=val_data, tokenizer=tokenizer)

    # Initialize model
    model = GliZNetModel(
        model_name=model_args.model_name,
        hidden_size=model_args.hidden_size,
        similarity_metric=model_args.similarity_metric,
    )

    # Create output directory
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        processing_class=tokenizer.tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        compute_metrics=compute_metrics,
    )

    # evaluate the model before training
    logger.info("Evaluating model before training...")
    eval_results = trainer.evaluate()
    logger.info(f"Initial evaluation results: {eval_results}")

    # Start training
    logger.info("Starting training with Transformers Trainer...")
    trainer.train()

    # Save the final model
    trainer.save_model(training_args.output_dir)

    # Also save in the legacy format for compatibility
    if model_args.save_path:
        os.makedirs(os.path.dirname(model_args.save_path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "training_args": training_args,
            },
            model_args.save_path,
        )
        logger.info(f"Legacy model saved at {model_args.save_path}")

    logger.info("Training complete")


if __name__ == "__main__":
    main()
