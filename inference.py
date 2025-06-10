import json
import torch
import numpy as np
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from loguru import logger
import os


class ZeroShotInference:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        with open(os.path.join(model_dir, 'training_config.json'), 'r') as f:
            self.config = json.load(f)
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        
        # Load all possible labels (optional, for suggestions)
        with open(os.path.join(model_dir, 'all_labels.json'), 'r') as f:
            self.all_labels = json.load(f)
            
        logger.success(f"Model loaded successfully. Known labels: {len(self.all_labels)}")
    
    def predict_single(self, text: str, candidate_labels: List[str], 
                      threshold: float = 0.5) -> Dict[str, any]:
        """
        Predict if text matches each of the candidate labels.
        
        Args:
            text: Input text to classify
            candidate_labels: List of potential labels to check
            threshold: Threshold for binary classification
            
        Returns:
            Dictionary with predictions and scores
        """
        predictions = []
        
        with torch.no_grad():
            for label in candidate_labels:
                # Create cross-encoder input: text + label
                encoding = self.tokenizer(
                    text,
                    label,
                    truncation=True,
                    padding='max_length',
                    max_length=self.config['max_length'],
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # Get model prediction
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Convert to probability (probability of match)
                probs = torch.softmax(logits, dim=-1)
                match_prob = probs[0, 1].item()  # Probability of class 1 (match)
                
                predictions.append({
                    'label': label,
                    'score': match_prob,
                    'prediction': match_prob > threshold
                })
        
        # Sort by score (highest first)
        predictions.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'text': text,
            'predictions': predictions,
            'matched_labels': [p['label'] for p in predictions if p['prediction']],
            'top_label': predictions[0]['label'] if predictions else None,
            'top_score': predictions[0]['score'] if predictions else 0.0
        }
    
    def predict_batch(self, texts: List[str], candidate_labels: List[str], 
                     threshold: float = 0.5) -> List[Dict[str, any]]:
        """Predict labels for multiple texts."""
        return [self.predict_single(text, candidate_labels, threshold) for text in texts]
    
    def suggest_labels(self, text: str, top_k: int = 10, threshold: float = 0.3) -> List[Tuple[str, float]]:
        """
        Suggest most likely labels from the training vocabulary.
        
        Args:
            text: Input text
            top_k: Number of top suggestions to return
            threshold: Minimum score threshold
            
        Returns:
            List of (label, score) tuples
        """
        if not self.all_labels:
            logger.warning("No label vocabulary available for suggestions")
            return []
        
        result = self.predict_single(text, self.all_labels, threshold)
        suggestions = [(p['label'], p['score']) for p in result['predictions'] 
                      if p['score'] > threshold]
        
        return suggestions[:top_k]
    
    def classify_with_hypothesis(self, text: str, hypothesis_template: str = "This text is about {}") -> Dict[str, any]:
        """
        Zero-shot classification using hypothesis template.
        
        Args:
            text: Input text
            hypothesis_template: Template with {} placeholder for labels
            
        Returns:
            Classification results
        """
        if not self.all_labels:
            logger.warning("No label vocabulary available")
            return {'text': text, 'predictions': []}
        
        # Create hypotheses for each label
        hypotheses = [hypothesis_template.format(label) for label in self.all_labels]
        
        predictions = []
        with torch.no_grad():
            for label, hypothesis in zip(self.all_labels, hypotheses):
                encoding = self.tokenizer(
                    text,
                    hypothesis,
                    truncation=True,
                    padding='max_length',
                    max_length=self.config['max_length'],
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                entailment_prob = probs[0, 1].item()
                
                predictions.append({
                    'label': label,
                    'hypothesis': hypothesis,
                    'score': entailment_prob
                })
        
        predictions.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'text': text,
            'predictions': predictions,
            'top_prediction': predictions[0] if predictions else None
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Zero-shot classification inference")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing trained model")
    parser.add_argument("--text", type=str, help="Text to classify")
    parser.add_argument("--labels", type=str, nargs='+', help="Candidate labels to check")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--suggest", action='store_true', help="Use suggestion mode with training labels")
    parser.add_argument("--hypothesis", action='store_true', help="Use hypothesis template mode")
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference = ZeroShotInference(args.model_dir)
    
    if args.text:
        if args.suggest:
            # Suggestion mode
            suggestions = inference.suggest_labels(args.text, top_k=10, threshold=0.3)
            print(f"\nText: {args.text}")
            print("\nSuggested Labels:")
            for label, score in suggestions:
                print(f"  {label}: {score:.4f}")
                
        elif args.hypothesis:
            # Hypothesis mode
            result = inference.classify_with_hypothesis(args.text)
            print(f"\nText: {result['text']}")
            print(f"\nTop Predictions:")
            for pred in result['predictions'][:10]:
                print(f"  {pred['label']}: {pred['score']:.4f}")
                
        elif args.labels:
            # Direct classification
            result = inference.predict_single(args.text, args.labels, args.threshold)
            print(f"\nText: {result['text']}")
            print(f"\nPredictions:")
            for pred in result['predictions']:
                status = "✓" if pred['prediction'] else "✗"
                print(f"  {status} {pred['label']}: {pred['score']:.4f}")
                
            if result['matched_labels']:
                print(f"\nMatched Labels: {', '.join(result['matched_labels'])}")
        else:
            print("Please provide --labels for classification or use --suggest/--hypothesis mode")
    else:
        # Interactive mode
        print("Interactive Zero-Shot Classification")
        print("Commands:")
        print("  classify <text> | <label1,label2,...>  - Classify with specific labels")
        print("  suggest <text>                         - Get label suggestions")
        print("  hypothesis <text>                      - Use hypothesis template")
        print("  quit                                   - Exit")
        
        while True:
            try:
                command = input("\n> ").strip()
                if command.lower() in ['quit', 'exit', 'q']:
                    break
                
                if command.startswith('classify '):
                    parts = command[9:].split(' | ')
                    if len(parts) == 2:
                        text, labels_str = parts
                        labels = [l.strip() for l in labels_str.split(',')]
                        result = inference.predict_single(text.strip(), labels, args.threshold)
                        
                        print(f"\nPredictions for: {result['text']}")
                        for pred in result['predictions']:
                            status = "✓" if pred['prediction'] else "✗"
                            print(f"  {status} {pred['label']}: {pred['score']:.4f}")
                    else:
                        print("Format: classify <text> | <label1,label2,...>")
                
                elif command.startswith('suggest '):
                    text = command[8:].strip()
                    suggestions = inference.suggest_labels(text, top_k=10)
                    print(f"\nSuggestions for: {text}")
                    for label, score in suggestions:
                        print(f"  {label}: {score:.4f}")
                
                elif command.startswith('hypothesis '):
                    text = command[11:].strip()
                    result = inference.classify_with_hypothesis(text)
                    print(f"\nHypothesis classification for: {text}")
                    for pred in result['predictions'][:5]:
                        print(f"  {pred['label']}: {pred['score']:.4f}")
                
                else:
                    print("Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()
