"""Generate synthetic training data from ArXiv abstracts using LiteLLM with Ollama."""

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import Dataset, load_dataset
from loguru import logger
from tqdm import tqdm

from generation_utils import generate_sample


def process_single_abstract(
    idx: int,
    dataset,
    model: str,
    api_base: str,
    output_dir: str,
    min_abstract_length: int,
) -> int:
    """Process a single abstract and save results to a file.

    Args:
        idx: Dataset index
        dataset: HuggingFace dataset
        model: Ollama model to use
        api_base: Ollama API base URL
        output_dir: Output directory for files
        min_abstract_length: Minimum abstract length

    Returns:
        Number of samples generated (1 or 0)
    """
    abstract = dataset[idx]["abstract"]

    # Skip very short abstracts
    if len(abstract) < min_abstract_length:
        return 0

    # Generate synthetic sample (single sentence)
    sample = generate_sample(
        text=abstract,
        model=model,
        api_base=api_base,
    )

    if not sample:
        return 0

    # Prepare output data
    output_data = {
        "source_arxiv_id": dataset[idx].get("arxiv_id", ""),
        "source_subject": dataset[idx].get("primary_subject", ""),
        "source_abstract": abstract,
        "sentence": sample.sentence,
        "labels": sample.labels,
        "not_labels": sample.not_labels,
    }

    # Save to individual file
    arxiv_id = dataset[idx].get("arxiv_id", f"sample_{idx}").replace("/", "_")
    output_file = os.path.join(output_dir, f"{arxiv_id}.json")

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    return 1


def generate_dataset_from_abstracts(
    dataset,
    num_abstracts: int = 100,
    model: str = "ollama/qwen2.5:14b",
    output_dir: str = "arxiv_synthetic_data",
    api_base: str = "http://localhost:11434",
    min_abstract_length: int = 100,
    num_workers: int = 2,
) -> int:
    """Generate training dataset from multiple abstracts in parallel.

    Args:
        dataset: HuggingFace dataset with 'abstract' field
        num_abstracts: Number of abstracts to process
        model: Ollama model to use
        output_dir: Output directory for JSON files
        api_base: Ollama API base URL
        min_abstract_length: Minimum abstract length to process
        num_workers: Number of parallel workers

    Returns:
        Total number of samples generated
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    if not num_abstracts or num_abstracts < 0:
        num_abstracts = len(dataset)

    # Randomly sample abstracts from the dataset
    indices = random.sample(range(len(dataset)), min(num_abstracts, len(dataset)))

    logger.info(f"Starting parallel generation from {num_abstracts} abstracts")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model: {model}")
    logger.info(f"Workers: {num_workers}")

    total_samples = 0

    # Process abstracts in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(
                process_single_abstract,
                idx,
                dataset,
                model,
                api_base,
                output_dir,
                min_abstract_length,
            ): idx
            for idx in indices
        }

        # Track progress with tqdm
        with tqdm(total=len(indices), desc="Processing abstracts") as pbar:
            for future in as_completed(future_to_idx):
                try:
                    num_samples = future.result()
                    total_samples += num_samples
                    pbar.update(1)
                    pbar.set_postfix({"total_samples": total_samples})
                except Exception as e:
                    idx = future_to_idx[future]
                    logger.error(f"Error processing abstract {idx}: {e}")
                    pbar.update(1)

    logger.info(f"✓ Generated {total_samples} samples from {len(indices)} abstracts")
    logger.info(f"✓ Saved to directory: {output_dir}")

    return total_samples


def save_as_hf_dataset(json_dir: str, output_dir: str) -> None:
    """Convert JSON files to HuggingFace Dataset format.

    Args:
        json_dir: Directory containing JSON files
        output_dir: Directory to save HuggingFace dataset
    """
    data = []

    # Read all JSON files from directory
    for filename in os.listdir(json_dir):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(json_dir, filename)
        with open(filepath, "r") as f:
            file_data = json.load(f)

            # Extract single sample from each file
            data.append(
                {
                    "text": file_data["sentence"],
                    "labels": file_data["labels"],
                    "not_labels": file_data["not_labels"],
                }
            )

    hf_dataset = Dataset.from_list(data)
    hf_dataset.save_to_disk(output_dir)
    logger.info(
        f"✓ Saved HuggingFace dataset to '{output_dir}' with {len(hf_dataset)} samples"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data from ArXiv abstracts"
    )
    parser.add_argument(
        "--num_abstracts",
        type=int,
        default=-1,
        help="Number of abstracts to process",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ollama/nemotron-3-nano",
        help="Ollama model to use (e.g., ollama/qwen2.5:14b, ollama/llama3.1:8b)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="arxiv_synthetic_data",
        help="Output directory for JSON files",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default="http://localhost:11434",
        help="Ollama API base URL",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="nick007x/arxiv-papers",
        help="HuggingFace dataset to load",
    )
    parser.add_argument(
        "--min_abstract_length",
        type=int,
        default=100,
        help="Minimum abstract length to process",
    )
    parser.add_argument(
        "--save_hf_dataset",
        action="store_true",
        help="Save output as HuggingFace dataset",
    )
    parser.add_argument(
        "--hf_output_dir",
        type=str,
        default="arxiv_synthetic_training_data",
        help="Directory to save HuggingFace dataset",
    )

    args = parser.parse_args()

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset)

    # Generate synthetic data
    generate_dataset_from_abstracts(
        dataset=ds["train"],
        num_abstracts=args.num_abstracts,
        output_dir=args.output_dir,
        model=args.model,
        api_base=args.api_base,
        min_abstract_length=args.min_abstract_length,
        num_workers=args.num_workers,
    )

    # Optionally save as HuggingFace dataset
    if args.save_hf_dataset:
        save_as_hf_dataset(args.output_dir, args.hf_output_dir)


if __name__ == "__main__":
    main()
