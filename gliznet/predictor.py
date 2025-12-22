from typing import Dict, List, Literal, Optional, Union

import torch

from gliznet.model import GliZNetForSequenceClassification
from gliznet.tokenizer import GliZNETTokenizer


class ZeroShotClassificationPipeline:
    """Zero-shot classification pipeline for GliZNet.
    
    Inspired by GLiClass, provides a simple interface for zero-shot text classification.
    
    Example:
        >>> from gliznet import GliZNetForSequenceClassification, ZeroShotClassificationPipeline
        >>> from gliznet.tokenizer import GliZNETTokenizer
        >>> 
        >>> model = GliZNetForSequenceClassification.from_pretrained("your-model")
        >>> tokenizer = GliZNETTokenizer.from_pretrained("your-model")
        >>> 
        >>> pipeline = ZeroShotClassificationPipeline(
        ...     model, tokenizer, classification_type='multi-label', device='cuda:0'
        ... )
        >>> 
        >>> text = "One day I will see the world!"
        >>> labels = ["travel", "dreams", "sport", "science", "politics"]
        >>> results = pipeline(text, labels, threshold=0.5)
        >>> 
        >>> for result in results[0]:
        ...     print(f"{result['label']} => {result['score']:.3f}")
    """

    def __init__(
        self,
        model: GliZNetForSequenceClassification,
        tokenizer: GliZNETTokenizer,
        classification_type: Literal["multi-label", "multi-class"] = "multi-label",
        device: Optional[str] = None,
    ):
        """Initialize the classification pipeline.

        Args:
            model: GliZNet model for classification
            tokenizer: Tokenizer for the model
            classification_type: Type of classification ('multi-label' or 'multi-class')
            device: Device to run inference on ('cpu', 'cuda', 'cuda:0', etc.)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.classification_type = classification_type
        
        # Set device
        if device is not None:
            self.device = torch.device(device)
            self.model = self.model.to(self.device)
        else:
            self.device = next(self.model.parameters()).device
        
        self.model.eval()
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        classification_type: Literal["multi-label", "multi-class"] = "multi-label",
        device: Optional[str] = None,
        **kwargs,
    ) -> "ZeroShotClassificationPipeline":
        """Load a pipeline from a pretrained model.
        
        Args:
            model_name_or_path: Path or identifier of the pretrained model
            classification_type: Type of classification ('multi-label' or 'multi-class')
            device: Device to run inference on
            **kwargs: Additional arguments for model loading
            
        Returns:
            Initialized pipeline
        """
        model, tokenizer = GliZNetForSequenceClassification.from_pretrained_with_tokenizer(
            model_name_or_path, **kwargs
        )
        return cls(model, tokenizer, classification_type=classification_type, device=device)

    def __call__(
        self,
        text: Union[str, List[str]],
        labels: Union[List[str], List[List[str]]],
        threshold: Optional[float] = None,
    ) -> List[List[Dict[str, Union[str, float]]]]:
        """Classify text(s) with given label(s).
        
        Args:
            text: Single text or list of texts to classify
            labels: Single label list or list of label lists (one per text)
            threshold: Minimum score threshold for filtering results (optional)
            
        Returns:
            List of predictions, where each prediction is a list of dicts with 'label' and 'score'
            
        Example:
            >>> results = pipeline("I love this movie!", ["positive", "negative"])
            >>> print(results[0])
            [{'label': 'positive', 'score': 0.95}, {'label': 'negative', 'score': 0.05}]
        """
        # Normalize inputs
        is_single = isinstance(text, str)
        if is_single:
            texts = [text]
            if isinstance(labels[0], str):
                all_labels = [labels]
            else:
                all_labels = labels
        else:
            texts = text
            if isinstance(labels[0], str):
                # Same labels for all texts
                all_labels = [labels] * len(texts)
            else:
                all_labels = labels
        
        # Get predictions
        results = self._predict_batch(texts, all_labels, threshold)
        
        return results if not is_single else results

    @torch.inference_mode()
    def _predict_batch(
        self,
        texts: List[str],
        all_labels: List[List[str]],
        threshold: Optional[float] = None,
    ) -> List[List[Dict[str, Union[str, float]]]]:
        """Internal batch prediction method."""
        # Tokenize
        batch = self.tokenizer(
            [(text, labels) for text, labels in zip(texts, all_labels)]
        )
        
        # Move to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        lmask = batch["lmask"].to(self.device)
        
        # Get model predictions
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            lmask=lmask,
            return_stats=True,
            return_dict=True,
        )
        
        # Process outputs
        logits = outputs.logits.squeeze(-1)
        batch_indices = outputs.batch_indices
        label_ids = outputs.label_ids
        
        # Apply activation based on classification type
        if self.classification_type == "multi-label":
            scores = torch.sigmoid(logits)
        else:  # multi-class
            scores = logits
        
        # Group by batch
        batch_size = len(texts)
        results = []
        
        scores_list = scores.cpu().tolist()
        batch_list = batch_indices.cpu().tolist()
        label_list = label_ids.cpu().tolist()
        
        for batch_id in range(batch_size):
            # Collect scores for this batch
            batch_scores = [
                (label_id, score)
                for b, label_id, score in zip(batch_list, label_list, scores_list)
                if b == batch_id
            ]
            
            if batch_scores:
                # Sort by label_id to match original label order
                batch_scores.sort(key=lambda x: x[0])
                label_scores = [score for _, score in batch_scores]
                
                # Apply softmax for multi-class
                if self.classification_type == "multi-class":
                    label_scores = torch.softmax(
                        torch.tensor(label_scores), dim=0
                    ).tolist()
                
                # Build result dicts
                batch_labels = all_labels[batch_id]
                batch_results = [
                    {"label": label, "score": score}
                    for label, score in zip(batch_labels, label_scores)
                ]
                
                # Apply threshold filter if specified
                if threshold is not None:
                    batch_results = [
                        r for r in batch_results if r["score"] >= threshold
                    ]
                
                # Sort by score (descending)
                batch_results.sort(key=lambda x: x["score"], reverse=True)
                
                results.append(batch_results)
            else:
                results.append([])
        
        return results
