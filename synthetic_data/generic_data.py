import json
import os
import random
from typing import Dict, List

from llm_clients import BaseLLMClient, create_llm_client
from llm_output_parser import parse_json
from loguru import logger
from prompts import generate_prompt
from tqdm import tqdm


class DatasetGenerator:
    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client
        self.all_topics = open("synthetic_data/all_subjects.txt").read().splitlines()
        logger.info(
            "Initialized DatasetGenerator with LLM client: "
            + str(len(self.all_topics))
            + " topics loaded"
        )

    def _create_prompt(self, num_samples: int, min_labels: int, max_labels: int) -> str:
        """Create the prompt for generating synthetic data."""
        topics = "\n".join(random.choices(self.all_topics, k=3))
        return (
            generate_prompt(
                num_samples=num_samples,
                min_labels=min_labels,
                max_labels=max_labels,
                topics=topics,
            ),
            topics,
        )

    def generate_dataset(
        self, num_samples: int, min_labels: int = 1, max_labels: int = 5
    ) -> List[Dict[str, List[str]]]:
        """Generate a single batch of synthetic data."""
        prompt, topics = self._create_prompt(num_samples, min_labels, max_labels)

        # LLM call - clearly separated
        raw_response = self.llm_client.generate_text(prompt)

        # Parse response
        try:
            data = parse_json(raw_response)
        except Exception:
            data = []
        data = data if isinstance(data, list) else []
        return {"topics": topics, "data": data}

    def generate_large_dataset(
        self, total_samples: int, batch_size: int = 50, output_dir: str = "batches"
    ) -> List[Dict[str, List[str]]]:
        """Generate large dataset in batches with progress tracking."""
        os.makedirs(output_dir, exist_ok=True)

        all_data = []
        batches = (total_samples + batch_size - 1) // batch_size

        # Use tqdm for progress tracking
        with tqdm(total=batches, desc="Generating batches", unit="batch") as pbar:
            for batch_num in range(batches):
                current_batch_size = min(batch_size, total_samples - len(all_data))
                batch_file = os.path.join(output_dir, f"batch_{batch_num + 1:06d}.json")

                # Update progress bar description
                pbar.set_description(
                    f"Batch {batch_num + 1}/{batches} ({current_batch_size} samples)"
                )

                # Generate batch data
                batch_data = self.generate_dataset(current_batch_size)

                # Save batch immediately
                with open(batch_file, "w") as f:
                    json.dump(batch_data, f, indent=2)

                all_data.extend(batch_data["data"])

                # Update progress bar
                pbar.set_postfix(
                    {
                        "samples": len(all_data),
                        "file": f"batch_{batch_num + 1:06d}.json",
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
        "--provider",
        type=str,
        choices=["groq", "ollama"],
        default="ollama",
        help="LLM provider to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cogito:latest",
        help="Model name to use",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for cloud providers (optional if set in env)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100_000,
        help="Number of synthetic samples to generate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=15,
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
        default="data/batches_cogito",
        help="Directory to save batch files",
    )

    args = parser.parse_args()

    logger.info(f"Starting synthetic data generation with {args.provider} provider")
    logger.info(f"Model: {args.model}")
    logger.info(f"Target samples: {args.num_samples}, Batch size: {args.batch_size}")

    # Create LLM client
    try:
        llm_client = create_llm_client(args.provider, args.model, args.api_key)
    except Exception as e:
        logger.error(f"Failed to create LLM client: {e}")
        return

    # Create dataset generator
    generator = DatasetGenerator(llm_client)

    # Generate data
    if args.num_samples > args.batch_size:
        logger.info("Using batch processing for large dataset")
        dataset = generator.generate_large_dataset(
            args.num_samples, args.batch_size, args.batch_dir
        )
    else:
        logger.info("Generating small dataset in single batch")
        dataset = generator.generate_dataset(
            args.num_samples, args.min_labels, args.max_labels
        )

    # Save final dataset
    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2)

    logger.success(f"Generated {len(dataset)} samples and saved to {args.output}")


if __name__ == "__main__":
    main()
