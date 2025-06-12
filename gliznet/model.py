import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FZeroNet(nn.Module):
    """
    Zero-shot classification model inspired by GLiNER.
    
    Architecture:
    1. BERT encoder to get contextualized embeddings
    2. Extract CLS token as text representation
    3. Extract first token of each label as label representation
    4. Compute similarity between text and label representations
    5. Apply sigmoid for multi-label classification
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        hidden_size: Optional[int] = None,
        dropout_rate: float = 0.1,
        similarity_metric: str = "cosine",  # "cosine", "dot", "bilinear"
        temperature: float = 1.0,
    ):
        """
        Initialize FZeroNet model.
        
        Args:
            model_name: HuggingFace model name for the base encoder
            hidden_size: Hidden size for projections (default: same as encoder)
            dropout_rate: Dropout rate for regularization
            similarity_metric: Method to compute similarity ("cosine", "dot", "bilinear")
            temperature: Temperature scaling for logits
        """
        super().__init__()
        
        # Load pre-trained BERT model
        self.encoder = AutoModel.from_pretrained(model_name)
        self.config = self.encoder.config
        
        # Set hidden size
        self.hidden_size = hidden_size or self.config.hidden_size
        self.similarity_metric = similarity_metric
        self.temperature = temperature
        
        # Projection layers for text and label representations
        self.text_projection = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.hidden_size),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        self.label_projection = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.hidden_size),
            nn.Dropout(dropout_rate),
            nn.ReLU(), 
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # For bilinear similarity
        if similarity_metric == "bilinear":
            self.bilinear = nn.Bilinear(self.hidden_size, self.hidden_size, 1)
        
        # Loss function
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        logger.info(f"Initialized FZeroNet with {model_name}")
        logger.info(f"Hidden size: {self.hidden_size}")
        logger.info(f"Similarity metric: {similarity_metric}")
    
    def get_label_positions(self, label_mask: torch.Tensor) -> List[List[int]]:
        """
        Extract the first token position for each label from the label mask.
        
        Args:
            label_mask: Boolean tensor of shape (batch_size, seq_len) indicating label positions
            
        Returns:
            List of lists containing first token positions for each label in each sample
        """
        batch_size, seq_len = label_mask.shape
        label_positions = []
        
        for batch_idx in range(batch_size):
            positions = []
            mask = label_mask[batch_idx]
            
            # Find contiguous label regions
            in_label = False
            for pos in range(seq_len):
                if mask[pos] and not in_label:
                    # Start of a new label
                    positions.append(pos)
                    in_label = True
                elif not mask[pos] and in_label:
                    # End of current label
                    in_label = False
            
            label_positions.append(positions)
        
        return label_positions
    
    def compute_similarity(
        self, 
        text_repr: torch.Tensor, 
        label_repr: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity between text and label representations.
        
        Args:
            text_repr: Text representations of shape (batch_size, hidden_size)
            label_repr: Label representations of shape (batch_size, num_labels, hidden_size)
            
        Returns:
            Similarity scores of shape (batch_size, num_labels)
        """
        if self.similarity_metric == "cosine":
            # Normalize vectors
            text_norm = F.normalize(text_repr, p=2, dim=-1)  # (batch_size, hidden_size)
            label_norm = F.normalize(label_repr, p=2, dim=-1)  # (batch_size, num_labels, hidden_size)
            
            # Compute cosine similarity
            # text_norm: (batch_size, 1, hidden_size)
            # label_norm: (batch_size, num_labels, hidden_size)
            similarity = torch.bmm(
                text_norm.unsqueeze(1), 
                label_norm.transpose(-1, -2)
            ).squeeze(1)  # (batch_size, num_labels)
            
        elif self.similarity_metric == "dot":
            # Simple dot product
            similarity = torch.bmm(
                text_repr.unsqueeze(1),
                label_repr.transpose(-1, -2)
            ).squeeze(1)
            
        elif self.similarity_metric == "bilinear":
            # Bilinear similarity
            batch_size, num_labels, hidden_size = label_repr.shape
            
            # Expand text_repr to match label_repr
            text_expanded = text_repr.unsqueeze(1).expand(-1, num_labels, -1)
            text_flat = text_expanded.reshape(-1, hidden_size)
            label_flat = label_repr.reshape(-1, hidden_size)
            
            # Apply bilinear layer
            similarity_flat = self.bilinear(text_flat, label_flat).squeeze(-1)
            similarity = similarity_flat.reshape(batch_size, num_labels)
            
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        # Apply temperature scaling
        return similarity / self.temperature
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label_mask: torch.Tensor,
        labels: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of FZeroNet.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            label_mask: Boolean mask indicating label positions (batch_size, seq_len)
            labels: List of ground truth label tensors (variable size per sample)
            
        Returns:
            Dictionary containing logits, loss (if labels provided), and representations
        """
        batch_size = input_ids.shape[0]
        
        # Get encoder outputs
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Extract hidden states
        hidden_states = encoder_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # Extract CLS token representation (first token after padding)
        # Find the first non-padding token (should be CLS)
        cls_positions = []
        for batch_idx in range(batch_size):
            mask = attention_mask[batch_idx]
            first_token_pos = torch.nonzero(mask, as_tuple=True)[0][0].item()
            cls_positions.append(first_token_pos)
        
        # Extract CLS representations
        cls_repr = torch.stack([
            hidden_states[i, pos] for i, pos in enumerate(cls_positions)
        ])  # (batch_size, hidden_size)
        
        # Project text representation
        text_repr = self.text_projection(cls_repr)  # (batch_size, hidden_size)
        
        # Extract label representations (first token of each label)
        label_positions = self.get_label_positions(label_mask)
        
        # For zero-shot, we process each sample individually with its own labels
        outputs_list = []
        total_loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
        
        for batch_idx in range(batch_size):
            positions = label_positions[batch_idx]
            num_labels = len(positions)
            
            if num_labels == 0:
                # No labels for this sample
                sample_logits = torch.zeros((0,), device=input_ids.device)
            else:
                # Extract label representations for this sample
                label_repr = torch.stack([
                    hidden_states[batch_idx, pos] for pos in positions
                ])  # (num_labels, hidden_size)
                
                # Project label representations
                label_repr_proj = self.label_projection(label_repr)  # (num_labels, hidden_size)
                
                # Compute similarity scores for this sample
                text_repr_sample = text_repr[batch_idx:batch_idx+1]  # (1, hidden_size)
                label_repr_sample = label_repr_proj.unsqueeze(0)  # (1, num_labels, hidden_size)
                
                sample_logits = self.compute_similarity(text_repr_sample, label_repr_sample).squeeze(0)  # (num_labels,)
            
            outputs_list.append({
                "logits": sample_logits,
                "num_labels": num_labels
            })
            
            # Compute loss for this sample if labels provided
            if labels is not None and batch_idx < len(labels):
                sample_labels = labels[batch_idx]
                if len(sample_labels) == len(sample_logits) and len(sample_logits) > 0:
                    sample_loss = self.loss_fn(sample_logits, sample_labels.float())
                    total_loss = total_loss + sample_loss
        
        # Return structured outputs for variable-size labels
        outputs = {
            "outputs_list": outputs_list,  # List of per-sample outputs
            "text_representations": text_repr,
        }
        
        if labels is not None:
            outputs["loss"] = total_loss / batch_size
        return outputs
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label_mask: torch.Tensor,
        threshold: float = 0.5,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Make predictions with the model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            label_mask: Boolean mask indicating label positions (batch_size, seq_len)
            threshold: Threshold for binary classification
            
        Returns:
            List of dictionaries containing per-sample predictions
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, label_mask)
            
            predictions_list = []
            for sample_output in outputs["outputs_list"]:
                sample_logits = sample_output["logits"]
                if len(sample_logits) > 0:
                    probabilities = torch.sigmoid(sample_logits)
                    predictions = (probabilities > threshold).long()
                else:
                    probabilities = torch.zeros((0,), device=input_ids.device)
                    predictions = torch.zeros((0,), device=input_ids.device, dtype=torch.long)
                
                predictions_list.append({
                    "predictions": predictions,
                    "probabilities": probabilities,
                    "logits": sample_logits,
                })
            
            return predictions_list


# Example usage and testing
if __name__ == "__main__":
    from tokenizer import ZeroShotClassificationTokenizer
    
    # Initialize model and tokenizer
    model = FZeroNet(model_name="bert-base-uncased", hidden_size=256)
    tokenizer = ZeroShotClassificationTokenizer()
    
    # Example data
    text = "A fascinating study published in the latest edition of Science Daily reveals that ancient humans used complex mathematical calculations for navigation and resource management."
    positive_labels = ["archaeological_findings", "scientific_discovery", "academic_publication"]
    negative_labels = ["historical_research", "science_fiction_story", "mathematics_education"]
    all_labels = positive_labels + negative_labels
    
    # Create ground truth (first 3 are positive, last 3 are negative)
    ground_truth = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.float32)
    
    # Tokenize
    inputs = tokenizer.tokenize_example(text, all_labels, return_tensors="pt")
    
    # Add batch dimension if needed
    for key in ["input_ids", "attention_mask", "label_mask"]:
        if inputs[key].dim() == 1:
            inputs[key] = inputs[key].unsqueeze(0)
    
    print("Input shapes:")
    print(f"Input IDs: {inputs['input_ids'].shape}")
    print(f"Attention mask: {inputs['attention_mask'].shape}")
    print(f"Label mask: {inputs['label_mask'].shape}")
    print(f"Ground truth: {ground_truth.shape}")
    
    # Forward pass
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        label_mask=inputs["label_mask"],
        labels=ground_truth
    )
    
    print("\nModel outputs:")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Logits: {outputs['logits']}")
    print(f"Loss: {outputs['loss']}")
    
    # Make predictions
    predictions = model.predict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        label_mask=inputs["label_mask"],
        threshold=0.5
    )
    
    print("\nPredictions:")
    print(f"Probabilities: {predictions['probabilities']}")
    print(f"Binary predictions: {predictions['predictions']}")
    print(f"Ground truth: {ground_truth}")