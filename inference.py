import json
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, util
from loguru import logger
import os


class ZeroShotInference:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir

        # Load trained SentenceTransformer model
        self.model = SentenceTransformer(model_dir)

        # Load configuration
        with open(os.path.join(model_dir, "training_config.json"), "r") as f:
            self.config = json.load(f)

        # Load all possible labels (for suggestions)
        with open(os.path.join(model_dir, "all_labels.json"), "r") as f:
            self.all_labels = json.load(f)

        logger.success(
            f"Model loaded successfully. Known labels: {len(self.all_labels)}"
        )

    def predict_similarity(
        self, texts: List[str], candidate_labels: List[str]
    ) -> List[Dict[str, any]]:
        """
        Predict similarity scores between texts and candidate labels.
        Uses cosine similarity between embeddings.
        """
        # Encode texts and labels
        text_embeddings = self.model.encode(
            texts, convert_to_tensor=True, show_progress_bar=False
        )
        label_embeddings = self.model.encode(
            candidate_labels, convert_to_tensor=True, show_progress_bar=False
        )

        # Compute cosine similarities
        similarities = util.cos_sim(text_embeddings, label_embeddings)

        results = []
        for i, text in enumerate(texts):
            text_similarities = similarities[i].cpu().numpy()

            # Create predictions with scores
            predictions = []
            for j, label in enumerate(candidate_labels):
                predictions.append(
                    {"label": label, "score": float(text_similarities[j])}
                )

            # Sort by score (highest first)
            predictions.sort(key=lambda x: x["score"], reverse=True)

            results.append(
                {
                    "text": text,
                    "predictions": predictions,
                    "top_label": predictions[0]["label"] if predictions else None,
                    "top_score": predictions[0]["score"] if predictions else 0.0,
                }
            )

        return results

    def predict_single(
        self, text: str, candidate_labels: List[str], top_k: int = None
    ) -> Dict[str, any]:
        """Predict labels for a single text."""
        result = self.predict_similarity([text], candidate_labels)[0]

        if top_k:
            result["predictions"] = result["predictions"][:top_k]

        return result

    def suggest_labels(
        self, text: str, top_k: int = 10, threshold: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Suggest most similar labels from the training vocabulary.
        """
        if not self.all_labels:
            logger.warning("No label vocabulary available for suggestions")
            return []

        result = self.predict_single(text, self.all_labels)
        suggestions = [
            (p["label"], p["score"])
            for p in result["predictions"]
            if p["score"] > threshold
        ]

        return suggestions[:top_k]

    def classify_with_threshold(
        self, text: str, candidate_labels: List[str], threshold: float = 0.5
    ) -> Dict[str, any]:
        """
        Classify text with a similarity threshold.
        """
        result = self.predict_single(text, candidate_labels)

        # Filter predictions above threshold
        matched_labels = [p for p in result["predictions"] if p["score"] > threshold]

        return {
            "text": text,
            "matched_labels": [p["label"] for p in matched_labels],
            "all_predictions": result["predictions"],
            "threshold": threshold,
            "num_matches": len(matched_labels),
        }

    def batch_classify(
        self, texts: List[str], candidate_labels: List[str], threshold: float = 0.5
    ) -> List[Dict[str, any]]:
        """Classify multiple texts efficiently."""
        return [
            self.classify_with_threshold(text, candidate_labels, threshold)
            for text in texts
        ]


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Zero-shot classification inference using similarity"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing trained model",
    )
    parser.add_argument("--text", type=str, help="Text to classify")
    parser.add_argument(
        "--labels", type=str, nargs="+", help="Candidate labels to check"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Similarity threshold"
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Return top-k predictions"
    )
    parser.add_argument(
        "--suggest",
        action="store_true",
        help="Use suggestion mode with training labels",
    )

    args = parser.parse_args()

    # Initialize inference engine
    inference = ZeroShotInference(args.model_dir)

    if args.text:
        if args.suggest:
            # Suggestion mode
            suggestions = inference.suggest_labels(
                args.text, top_k=args.top_k, threshold=0.3
            )
            print(f"\nText: {args.text}")
            print("\nSuggested Labels (similarity scores):")
            for label, score in suggestions:
                print(f"  {label}: {score:.4f}")

        elif args.labels:
            # Direct classification with threshold
            result = inference.classify_with_threshold(
                args.text, args.labels, args.threshold
            )
            print(f"\nText: {result['text']}")
            print(f"Threshold: {result['threshold']}")
            print("\nAll Predictions:")
            for pred in result["all_predictions"]:
                status = "✓" if pred["score"] > args.threshold else "✗"
                print(f"  {status} {pred['label']}: {pred['score']:.4f}")

            if result["matched_labels"]:
                print(f"\nMatched Labels: {', '.join(result['matched_labels'])}")
            else:
                print(f"\nNo matches above threshold {args.threshold}")
        else:
            print("Please provide --labels for classification or use --suggest mode")
    else:
        # Interactive mode
        print("Interactive Zero-Shot Classification (Similarity-based)")
        print("Commands:")
        print(
            "  classify <text> | <label1,label2,...>  - Classify with specific labels"
        )
        print("  suggest <text>                         - Get label suggestions")
        print("  quit                                   - Exit")

        while True:
            try:
                command = input("\n> ").strip()
                if command.lower() in ["quit", "exit", "q"]:
                    break

                if command.startswith("classify "):
                    parts = command[9:].split(" | ")
                    if len(parts) == 2:
                        text, labels_str = parts
                        labels = [l.strip() for l in labels_str.split(",")]
                        result = inference.classify_with_threshold(
                            text.strip(), labels, args.threshold
                        )

                        print(f"\nSimilarity scores for: {result['text']}")
                        for pred in result["all_predictions"]:
                            status = "✓" if pred["score"] > args.threshold else "✗"
                            print(f"  {status} {pred['label']}: {pred['score']:.4f}")

                        if result["matched_labels"]:
                            print(f"Matches: {', '.join(result['matched_labels'])}")
                    else:
                        print("Format: classify <text> | <label1,label2,...>")

                elif command.startswith("suggest "):
                    text = command[8:].strip()
                    suggestions = inference.suggest_labels(text, top_k=args.top_k)
                    print(f"\nSuggestions for: {text}")
                    for label, score in suggestions:
                        print(f"  {label}: {score:.4f}")

                else:
                    print("Unknown command. Type 'quit' to exit.")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()
