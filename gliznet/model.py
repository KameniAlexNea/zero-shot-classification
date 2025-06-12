import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class GliZNet(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        hidden_size: int = None,
        dropout_rate: float = 0.1,
        similarity_metric: str = "cosine",
        temperature: float = 1.0,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.config = self.encoder.config

        self.hidden_size = hidden_size or self.config.hidden_size
        self.similarity_metric = similarity_metric
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout_rate)

        # Optional projection
        if self.hidden_size != self.config.hidden_size:
            self.proj = nn.Linear(self.config.hidden_size, self.hidden_size)
        else:
            self.proj = nn.Identity()

        if similarity_metric == "bilinear":
            self.bilinear = nn.Bilinear(self.hidden_size, self.hidden_size, 1)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def compute_similarity(self, text_repr, label_repr):
        if self.similarity_metric == "cosine":
            text_norm = F.normalize(text_repr, dim=-1)
            label_norm = F.normalize(label_repr, dim=-1)
            sim = torch.bmm(text_norm.unsqueeze(1), label_norm.transpose(1, 2)).squeeze(1)
        elif self.similarity_metric == "dot":
            sim = torch.bmm(text_repr.unsqueeze(1), label_repr.transpose(1, 2)).squeeze(1)
        elif self.similarity_metric == "bilinear":
            batch_size, num_labels, _ = label_repr.size()
            text_exp = text_repr.unsqueeze(1).expand(-1, num_labels, -1)
            sim = self.bilinear(text_exp.reshape(-1, self.hidden_size), label_repr.reshape(-1, self.hidden_size)).squeeze(-1)
            sim = sim.view(batch_size, num_labels)
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")
        return sim / self.temperature

    def forward(self, input_ids, attention_mask, label_mask, labels=None):
        device = input_ids.device
        encoder_out = self.encoder(input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = self.dropout(encoder_out.last_hidden_state)
        hidden_proj = self.proj(hidden)

        # CLS-based text representation
        text_repr = hidden_proj[:, 0]  # (batch, hidden_size)

        batch_size = input_ids.size(0)
        outputs_list = []
        total_loss = torch.tensor(0.0, device=device)

        for i in range(batch_size):
            label_indices = torch.nonzero(label_mask[i], as_tuple=True)[0]
            if len(label_indices) == 0:
                logits = torch.zeros((0,), device=device)
            else:
                label_repr = hidden_proj[i][label_indices]  # (num_labels, hidden_size)
                logits = self.compute_similarity(text_repr[i].unsqueeze(0), label_repr.unsqueeze(0)).squeeze(0)

            outputs_list.append({"logits": logits, "num_labels": len(label_indices)})

            if labels is not None and i < len(labels):
                label_targets = labels[i]
                if len(label_targets) == len(logits) and len(logits) > 0:
                    loss = self.loss_fn(logits, label_targets.to(device))
                    total_loss += loss

        output = {"outputs_list": outputs_list, "text_representations": text_repr}
        if labels is not None:
            output["loss"] = total_loss / batch_size
        return output

    def predict(self, input_ids, attention_mask, label_mask, threshold=0.5):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, label_mask)
            results = []

            for sample in outputs["outputs_list"]:
                logits = sample["logits"]
                probs = torch.sigmoid(logits)
                preds = (probs > threshold).long()
                results.append({
                    "predictions": preds,
                    "probabilities": probs,
                    "logits": logits
                })
        return results



# Example usage and testing
if __name__ == "__main__":
    from tokenizer import GliZNETTokenizer
    
    # Initialize model and tokenizer
    model_name="bert-base-uncased"
    model = GliZNet(model_name=model_name, hidden_size=256)
    tokenizer = GliZNETTokenizer(model_name)
    
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