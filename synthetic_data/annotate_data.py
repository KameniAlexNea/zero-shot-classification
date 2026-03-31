import json
import os
import random
from datetime import datetime

import datasets
from dotenv import load_dotenv
from generation_utils import generate_sample
from tqdm import tqdm

load_dotenv()

query_col = None
# data = datasets.load_dataset("MongoDB/cosmopedia-wikihow-chunked", split="train")
# data = datasets.load_dataset("fancyzhx/ag_news", split="train")
text_col = "text"
# data = datasets.load_dataset("wikimedia/wikipedia", name="20231101.en", split="train")
text_col = "text"
# data = datasets.load_dataset("FutureMa/DramaBench", split="train")
text_col = "description"
# data = datasets.load_dataset("qiaojin/PubMedQA", name="pqa_unlabeled", split="train")
text_col = "long_answer"
query_col = "question"
# data = datasets.load_dataset("cmpatino/news-bias-detection-dataset", name="all", split="train")
text_col = "text"
query_col = None
# data = datasets.load_dataset("pietrolesci/nli_fever", split="train")
text_col = "hypothesis"
query_col = "premise"
# data = datasets.load_dataset("domenicrosati/TruthfulQA", split="train")
text_col = "Correct Answers"
query_col = "Question"
# data = datasets.load_dataset("MuskumPillerum/General-Knowledge", split="train")
text_col = "Answer"
query_col = "Question"
# data = datasets.load_dataset("gretelai/symptom_to_diagnosis", split="train")
text_col = "input_text"
query_col = "output_text"
# data = datasets.load_dataset("asuender/motivational-quotes", name="quotes", split="train")
text_col = "quote"
query_col = "author"
data = datasets.load_dataset("brando/small-c4-dataset", split="train")
text_col = "text"
query_col = None


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
    api_key=None,
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
    indices = random.sample(range(len(dataset)), min(total_samples, len(dataset)))

    print(f"Processing {total_samples} samples starting from index {start_index}")
    print(f"Batch size: {batch_size}")
    print(f"Model: {model}")
    print(f"Output directory: {output_dir}")

    # Create progress bar
    pbar = tqdm(
        indices[start_index : start_index + total_samples],
        desc="Processing samples",
        total=total_samples,
        unit="sample",
    )

    for i in pbar:
        if i >= len(dataset):
            break
        raw = dataset[i]

        text = raw[text_col]
        if text is None or text.strip() == "" or len(text) < 10:
            pbar.write(f"Skipping sample {i + 1} due to empty text")
            continue
        text = text[:5000]  # Truncate to first 5000 characters
        if query_col:
            text = f"{query_col}: {raw[query_col]}\n\n{text_col}: {text}"
        pbar.set_description(f"Processing sample {i + 1}...")

        generated_sample = generate_sample(
            text=text,
            model=model,
            api_base=api_base,
            api_key=api_key,
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
        batch_size=50,
        start_index=0,
        max_samples=1500,  # Set to None to process entire dataset
        # model="ollama/nemotron-3-nano",  # or "ollama/llama3.1:8b", "ollama/nemotron-3-nano"
        model="ollama/nemotron-3-nano:30b",  # or "ollama/llama3.1:8b", "ollama/nemotron-3-nano"
        # model="openai/gpt-oss-120b",  # or "ollama/llama3.1:8b", "ollama/nemotron-3-nano"
        # api_base="http://localhost:11434",
        api_base="https://ollama.com",
        # api_base="https://api.groq.com/openai/v1",
        output_dir=f"synthetic_data/small-c4_synthetic_data_nemotron3_nano_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        # api_key=os.environ.get("GROQ_API_KEY"),
        api_key=os.environ.get("OLLAMA_API_KEY_V2"),
    )
