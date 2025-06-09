import json
from typing import List, Dict
from ollama import Client
from llm_output_parser import parse_json


class OllamaClient:
    def __init__(self, model: str = "phi4:latest"):
        self.client = Client()
        self.model = model

    def generate_dataset(
        self, num_samples: int, min_labels: int = 1, max_labels: int = 5
    ) -> List[Dict[str, List[str]]]:
        import random

        # Randomize temperature and seed for diversity
        temperature = random.uniform(0.7, 0.9)
        seed = random.randint(1, 100000)

        prompt = f"""You are an expert data generator for machine learning classification tasks.

TASK: Generate exactly {num_samples} diverse text examples for zero-shot classification training.

WHAT IS A LABEL: A label is a category or classification that describes what the text is about, its purpose, tone, domain, or type. Labels help classify text into meaningful groups for machine learning.

Examples of good labels:
- Content type: "news_article", "product_review", "social_media_post", "email", "instruction"  
- Domain: "technology", "sports", "finance", "healthcare", "entertainment", "politics"
- Sentiment: "positive", "negative", "neutral", "complaint", "praise"
- Intent: "question", "request", "announcement", "opinion", "fact"
- Style: "formal", "casual", "technical", "promotional", "educational"

REQUIREMENTS:
- Create {num_samples} completely different sentences (5-25 words each)
- Mix topics: technology, business, sports, health, entertainment, science, politics, lifestyle, education
- Mix text types: statements, questions, reviews, news, instructions, opinions, social posts
- Mix writing styles: formal, casual, technical, promotional, conversational
- For each sentence, provide {min_labels}-{max_labels} relevant, specific labels
- Labels should be descriptive and useful for classification
- Ensure maximum diversity in both content and labels

OUTPUT FORMAT: Return as valid JSON array only:
[
  {{"sentence": "The new smartphone camera produces amazing low-light photos.", "labels": ["technology", "product_review", "positive", "consumer_electronics"]}},
  {{"sentence": "How do I reset my password for the company portal?", "labels": ["question", "technical_support", "workplace", "instruction_request"]}}
]

Generate exactly {num_samples} diverse entries:"""

        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            options={
                "seed": seed,
                "temperature": temperature,
            },
        )
        data = parse_json(response.text.strip())
        return data if isinstance(data, list) else []

    def generate_large_dataset(
        self, total_samples: int, batch_size: int = 50, output_dir: str = "batches"
    ) -> List[Dict[str, List[str]]]:
        import os

        os.makedirs(output_dir, exist_ok=True)

        all_data = []
        batches = (total_samples + batch_size - 1) // batch_size

        for batch_num in range(batches):
            current_batch_size = min(batch_size, total_samples - len(all_data))
            batch_file = os.path.join(output_dir, f"batch_{batch_num + 1:04d}.json")

            batch_data = self.generate_dataset(current_batch_size)

            # Save batch immediately
            with open(batch_file, "w") as f:
                json.dump(batch_data, f, indent=2)

            all_data.extend(batch_data)
            print(f"✓ Batch {batch_num + 1} saved to {batch_file}")

        return all_data


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic data for zero-shot text classification."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="phi4:latest",
        help="Name of the Ollama model to use",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of synthetic samples to generate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for large dataset generation (default: 50)",
    )
    parser.add_argument(
        "--min-labels",
        type=int,
        default=1,
        help="Minimum number of labels per sentence (default: 1)",
    )
    parser.add_argument(
        "--max-labels",
        type=int,
        default=5,
        help="Maximum number of labels per sentence (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="synthetic_data.json",
        help="Output JSON filepath",
    )
    parser.add_argument(
        "--batch-dir",
        type=str,
        default="batches",
        help="Directory to save batch files (default: batches)",
    )
    args = parser.parse_args()

    client = OllamaClient(model=args.model)

    if args.num_samples > args.batch_size:
        dataset = client.generate_large_dataset(
            args.num_samples, args.batch_size, args.batch_dir
        )
    else:
        dataset = client.generate_dataset(args.num_samples)

    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Generated {len(dataset)} samples and saved to {args.output}")


if __name__ == "__main__":
    main()
