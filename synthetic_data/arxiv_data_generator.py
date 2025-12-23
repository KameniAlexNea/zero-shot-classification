"""Generate synthetic training data from ArXiv abstracts."""

import json
import os
import random
from datetime import datetime

from datasets import load_dataset
from generation_utils import generate_sample
from tqdm import tqdm


def save_batch_to_json(batch_data, batch_number, output_dir):
    """Save a batch of processed data to JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{output_dir}/batch_{batch_number:04d}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(batch_data, f, indent=2, ensure_ascii=False)

    print(f"Saved batch {batch_number} with {len(batch_data)} samples to {filename}")


def process_dataset_in_batches(
    dataset,
    batch_size=10,
    start_index=0,
    max_samples=None,
    model="ollama/qwen2.5:14b",
    api_base="http://localhost:11434",
    output_dir="arxiv_synthetic_data",
    min_abstract_length=100,
):
    """Process dataset in batches and save to JSON files.

    Args:
        dataset: HuggingFace dataset with 'abstract' field
        batch_size: Number of samples per batch file
        start_index: Starting index in dataset
        max_samples: Maximum number of samples to process (None = all)
        model: Ollama model to use
        api_base: Ollama API base URL
        output_dir: Output directory for batch files
        min_abstract_length: Minimum abstract length to process
    """
    batch_data = []
    batch_number = start_index // batch_size + 1
    processed_count = 0

    # Determine how many samples to process
    total_samples = (
        len(dataset) if max_samples is None else min(max_samples, len(dataset))
    )

    # Randomly sample indices
    indices = random.sample(range(len(dataset)), min(total_samples, len(dataset)))

    print(f"Processing {total_samples} abstracts starting from index {start_index}")
    print(f"Batch size: {batch_size}")
    print(f"Model: {model}")
    print(f"Output directory: {output_dir}")

    # Create progress bar
    pbar = tqdm(
        indices[start_index : start_index + total_samples],
        desc="Processing abstracts",
        total=total_samples,
        unit="abstract",
    )

    for idx in pbar:
        if idx >= len(dataset):
            break

        abstract = dataset[idx]["abstract"]

        # Skip very short abstracts
        if len(abstract) < min_abstract_length:
            pbar.write(f"Skipping sample {idx} (too short)")
            continue

        pbar.set_description(f"Processing abstract {idx}...")

        generated_sample = generate_sample(
            text=abstract,
            model=model,
            api_base=api_base,
        )

        if generated_sample:
            sample = {
                "source_arxiv_id": dataset[idx].get("arxiv_id", ""),
                "source_subject": dataset[idx].get("primary_subject", ""),
                "source_abstract": abstract,
                "sentence": generated_sample.sentence,
                "labels": generated_sample.labels,
                "not_labels": generated_sample.not_labels,
            }
            batch_data.append(sample)
            processed_count += 1

            # Save batch when it reaches the batch size
            if len(batch_data) >= batch_size:
                save_batch_to_json(batch_data, batch_number, output_dir)
                batch_data = []
                batch_number += 1
                pbar.set_postfix(
                    batches_saved=batch_number - 1, processed=processed_count
                )
        else:
            pbar.write(f"Skipping sample {idx} due to generation failure")

    pbar.close()

    # Save remaining data if any
    if batch_data:
        save_batch_to_json(batch_data, batch_number, output_dir)

    print(f"✓ Processing complete! Total samples processed: {processed_count}")
    print(f"✓ Saved to directory: {output_dir}")


if __name__ == "__main__":
    # Load dataset
    print("Loading ArXiv dataset...")
    ds = load_dataset("nick007x/arxiv-papers")

    # Process abstracts in batches
    process_dataset_in_batches(
        dataset=ds["train"],
        batch_size=50,
        start_index=0,
        max_samples=1_000_000,
        model="ollama/nemotron-3-nano",
        api_base="http://localhost:11434",
        output_dir=f"synthetic_data/arxiv_synthetic_data_nemotron3_nano_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        min_abstract_length=100,
    )
