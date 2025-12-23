import json
import os
from datetime import datetime

import datasets
from generation_utils import generate_sample
from tqdm import tqdm

data = datasets.load_dataset("MongoDB/cosmopedia-wikihow-chunked", split="train")


def save_batch_to_json(
    batch_data, batch_number, output_dir="synthetic_data/annotated_batches"
):
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
    output_dir="synthetic_data/annotated_batches",
):
    """Process dataset in batches and save to JSON files.

    Args:
        dataset: HuggingFace dataset with 'text' field
        batch_size: Number of samples per batch file
        start_index: Starting index in dataset
        max_samples: Maximum number of samples to process (None = all)
        model: Ollama model to use
        api_base: Ollama API base URL
        output_dir: Output directory for batch files
    """
    batch_data = []
    batch_number = start_index // batch_size + 1
    processed_count = 0

    # Determine how many samples to process
    total_samples = (
        len(dataset) if max_samples is None else min(max_samples, len(dataset))
    )

    print(f"Processing {total_samples} samples starting from index {start_index}")
    print(f"Batch size: {batch_size}")
    print(f"Model: {model}")
    print(f"Output directory: {output_dir}")

    # Create progress bar
    pbar = tqdm(
        range(start_index, start_index + total_samples),
        desc="Processing samples",
        total=total_samples,
        unit="sample",
    )

    for i in pbar:
        if i >= len(dataset):
            break

        text = dataset[i]["text"]
        pbar.set_description(f"Processing sample {i + 1}...")

        generated_sample = generate_sample(
            text=text,
            model=model,
            api_base=api_base,
        )

        if generated_sample:
            sample = {
                "source_text": text,
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
            pbar.write(f"Skipping sample {i + 1} due to generation failure")

    pbar.close()

    # Save remaining data if any
    if batch_data:
        save_batch_to_json(batch_data, batch_number, output_dir)

    print(f"✓ Processing complete! Total samples processed: {processed_count}")
    print(f"✓ Saved to directory: {output_dir}")


if __name__ == "__main__":
    # Example usage: process first 100 samples in batches of 10
    process_dataset_in_batches(
        dataset=data,
        batch_size=5,
        start_index=0,
        max_samples=15,  # Set to None to process entire dataset
        # model="ollama/nemotron-3-nano",  # or "ollama/llama3.1:8b", "ollama/nemotron-3-nano"
        model="openai/gpt-oss-120b",  # or "ollama/llama3.1:8b", "ollama/nemotron-3-nano"
        # api_base="http://localhost:11434",
        api_base="https://api.groq.com/openai/v1",
        output_dir=f"synthetic_data/wikihow_synthetic_data_nemotron3_nano_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        api_key=os.environ.get("GROQ_API_KEY"),
    )
