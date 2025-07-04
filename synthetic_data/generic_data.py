import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
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

    def _create_prompt(self, num_samples: int) -> str:
        """Create the prompt for generating synthetic data."""
        topics = "\n".join(random.choices(self.all_topics, k=3))
        return (
            generate_prompt(
                num_samples=num_samples,
                topics=topics,
            ),
            topics,
        )

    def generate_dataset(self, num_samples: int) -> List[Dict[str, List[str]]]:
        """Generate a single batch of synthetic data."""
        prompt, topics = self._create_prompt(num_samples)

        # LLM call - clearly separated
        raw_response = self.llm_client.generate_text(prompt)

        # Parse response
        try:
            data = parse_json(raw_response)
        except Exception:
            data = []
        data = data if isinstance(data, list) else []
        return {"topics": topics, "data": data}

    def _save_batch_file(self, batch_data: Dict, batch_file: str) -> None:
        """Save batch data to file (used for parallel execution)."""
        with open(batch_file, "w") as f:
            json.dump(batch_data, f, indent=2)

    def generate_large_dataset(
        self, total_samples: int, batch_size: int = 50, output_dir: str = "batches"
    ) -> None:
        """Generate large dataset in batches with progress tracking and parallel saving."""
        os.makedirs(output_dir, exist_ok=True)

        total_generated = 0
        batches = (total_samples + batch_size - 1) // batch_size

        # Use ThreadPoolExecutor for parallel file saving
        with ThreadPoolExecutor(max_workers=4) as executor, tqdm(
            total=batches, desc="Generating batches", unit="batch"
        ) as pbar:
            save_futures = []

            for batch_num in range(batches):
                current_batch_size = min(batch_size, total_samples - total_generated)
                batch_file = os.path.join(output_dir, f"batch_{batch_num + 1:06d}.json")

                # Update progress bar description
                pbar.set_description(
                    f"Batch {batch_num + 1}/{batches} ({current_batch_size} samples)"
                )

                # Generate batch data
                batch_data = self.generate_dataset(current_batch_size)

                # Submit save task to thread pool (non-blocking)
                save_future = executor.submit(
                    self._save_batch_file, batch_data, batch_file
                )
                save_futures.append(save_future)

                total_generated += len(batch_data["data"])

                # Update progress bar
                pbar.set_postfix(
                    {
                        "samples": total_generated,
                        "file": f"batch_{batch_num + 1:06d}.json",
                    }
                )
                pbar.update(1)

            # Wait for all save operations to complete
            logger.info("Waiting for all file saves to complete...")
            for future in save_futures:
                future.result()  # This will raise any exceptions that occurred

        logger.success(
            f"Generated {total_generated} samples across {batches} batches in {output_dir}"
        )


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
        generator.generate_large_dataset(
            args.num_samples, args.batch_size, args.batch_dir
        )
    else:
        logger.info("Generating small dataset in single batch")
        batch_data = generator.generate_dataset(args.num_samples)

        # Save single batch file
        os.makedirs(args.batch_dir, exist_ok=True)
        batch_file = os.path.join(args.batch_dir, "batch_000001.json")
        with open(batch_file, "w") as f:
            json.dump(batch_data, f, indent=2)

        logger.success(
            f"Generated {len(batch_data['data'])} samples and saved to {batch_file}"
        )

    logger.success(f"All data saved to batch files in {args.batch_dir}")


if __name__ == "__main__":
    main()
