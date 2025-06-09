import json
from typing import List, Dict
from ollama import Client
from llm_output_parser import parse_json
from tqdm import tqdm
from loguru import logger
from prompts import generate_prompt


class OllamaClient:
    def __init__(self, model: str = "phi4:latest"):
        self.client = Client()
        self.model = model

    def _create_prompt(self, num_samples: int, min_labels: int, max_labels: int) -> str:
        """Create the prompt for generating synthetic data."""
        return generate_prompt(
            num_samples=num_samples,
            min_labels=min_labels,
            max_labels=max_labels,
        )

    def generate_dataset(
        self, num_samples: int, min_labels: int = 1, max_labels: int = 5
    ) -> List[Dict[str, List[str]]]:
        import random

        # Randomize temperature and seed for diversity
        temperature = random.uniform(0.7, 0.9)
        seed = random.randint(1, 100000)

        prompt = self._create_prompt(num_samples, min_labels, max_labels)

        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            options={
                "seed": seed,
                "temperature": temperature,
            },
        )
        data = parse_json(response.response.strip())
        return data if isinstance(data, list) else []

    def generate_large_dataset(
        self, total_samples: int, batch_size: int = 50, output_dir: str = "batches"
    ) -> List[Dict[str, List[str]]]:
        import os

        os.makedirs(output_dir, exist_ok=True)

        all_data = []
        batches = (total_samples + batch_size - 1) // batch_size

        # Use tqdm for progress tracking
        with tqdm(total=batches, desc="Generating batches", unit="batch") as pbar:
            for batch_num in range(batches):
                current_batch_size = min(batch_size, total_samples - len(all_data))
                batch_file = os.path.join(output_dir, f"batch_{batch_num + 1:04d}.json")

                # Update progress bar description
                pbar.set_description(
                    f"Batch {batch_num + 1}/{batches} ({current_batch_size} samples)"
                )

                batch_data = self.generate_dataset(current_batch_size)

                # Save batch immediately
                with open(batch_file, "w") as f:
                    json.dump(batch_data, f, indent=2)

                all_data.extend(batch_data)

                # Update progress bar
                pbar.set_postfix(
                    {
                        "samples": len(all_data),
                        "file": f"batch_{batch_num + 1:04d}.json",
                    }
                )
                pbar.update(1)

        logger.success(f"Generated {len(all_data)} samples across {batches} batches")
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
        default="data/synthetic_data.json",
        help="Output JSON filepath",
    )
    parser.add_argument(
        "--batch-dir",
        type=str,
        default="data/batches",
        help="Directory to save batch files (default: batches)",
    )
    args = parser.parse_args()

    logger.info(f"Starting synthetic data generation with model: {args.model}")
    logger.info(f"Target samples: {args.num_samples}, Batch size: {args.batch_size}")

    client = OllamaClient(model=args.model)

    if args.num_samples > args.batch_size:
        logger.info("Using batch processing for large dataset")
        dataset = client.generate_large_dataset(
            args.num_samples, args.batch_size, args.batch_dir
        )
    else:
        logger.info("Generating small dataset in single batch")
        dataset = client.generate_dataset(args.num_samples)

    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2)

    logger.success(f"Generated {len(dataset)} samples and saved to {args.output}")


if __name__ == "__main__":
    main()
