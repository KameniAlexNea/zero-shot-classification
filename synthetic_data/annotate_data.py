import ollama
import datasets
import json
import os
from tqdm import tqdm
from llm_output_parser import parse_json

data = datasets.load_dataset("sentence-transformers/wikihow", split="train")

PROMPT = """
You are an expert text classification assistant. Your task is to analyze a given text and generate comprehensive labels for zero-shot classification training.

**ANALYSIS REQUIREMENTS**:
- Analyze the text's content, context, tone, intent, and domain
- Consider nuanced aspects that require deep understanding
- Focus on descriptive labels that capture semantic meaning beyond keywords

**POSITIVE LABELS**:
- Generate 5-15 accurate, descriptive labels that summarize the text's content
- Include labels for: content type, domain/topic, tone, intent, style, target audience
- Use specific, meaningful labels rather than generic ones
- Examples: instructional_content, health_advice, step_by_step_guide, practical_tutorial

**HARD NEGATIVE LABELS**:
- Generate 5-15 challenging negative labels that are plausible but incorrect
- Should be semantically related but contextually wrong
- Require careful analysis to distinguish from correct labels
- Examples: academic_research, entertainment_content, product_advertisement

**OUTPUT FORMAT**:
Provide exactly 2 lines:
Line 1: Positive labels (comma-separated)
Line 2: Hard negative labels (comma-separated)

**Text to analyze**: "{text}"
"""


def generate_labels_for_text(text: str):
    """Generate positive and negative labels for a given text using Ollama."""
    try:
        response = ollama.chat(
            model="cogito",
            messages=[{"role": "user", "content": PROMPT.format(text=text)}],
        )

        content: str = response["message"]["content"].strip()
        lines = content.split("\n")

        if len(lines) >= 2:
            positive_labels = [
                label.strip()
                for label in lines[0].replace("Line 1: ", "").split(",")
                if label.replace("- ", "").strip()
            ]
            negative_labels = [
                label.strip()
                for label in lines[1].replace("Line 2: ", "").split(",")
                if label.replace("- ", "").strip()
            ]
            return positive_labels, negative_labels
        else:
            print(f"Unexpected response format: {content}")
            return [], []

    except Exception as e:
        print(f"Error generating labels: {e}")
        return [], []


def save_batch_to_json(batch_data, batch_number):
    """Save a batch of processed data to JSON file."""
    output_dir = "synthetic_data/annotated_batches"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{output_dir}/batch_{batch_number:04d}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(batch_data, f, indent=2, ensure_ascii=False)

    print(f"Saved batch {batch_number} with {len(batch_data)} samples to {filename}")


def process_dataset_in_batches(dataset, batch_size=10, start_index=0, max_samples=None):
    """Process dataset in batches and save to JSON files."""
    batch_data = []
    batch_number = start_index // batch_size + 1
    processed_count = 0

    # Determine how many samples to process
    total_samples = (
        len(dataset) if max_samples is None else min(max_samples, len(dataset))
    )

    print(f"Processing {total_samples} samples starting from index {start_index}")
    print(f"Batch size: {batch_size}")

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
        pbar.set_description(f"Processing sample {i + 1}: {text[:50]}...")

        positive_labels, negative_labels = generate_labels_for_text(text)

        if positive_labels and negative_labels:
            sample = {
                "text": text,
                "positive": positive_labels,
                "negative": negative_labels,
            }
            batch_data.append(sample)
            processed_count += 1

            # Save batch when it reaches the batch size
            if len(batch_data) >= batch_size:
                save_batch_to_json(batch_data, batch_number)
                batch_data = []
                batch_number += 1
                pbar.set_postfix(
                    batches_saved=batch_number - 1, processed=processed_count
                )
        else:
            pbar.write(f"Skipping sample {i + 1} due to label generation failure")

    pbar.close()

    # Save remaining data if any
    if batch_data:
        save_batch_to_json(batch_data, batch_number)

    print(f"Processing complete! Total samples processed: {processed_count}")


if __name__ == "__main__":
    # Example usage: process first 100 samples in batches of 10
    process_dataset_in_batches(
        dataset=data,
        batch_size=10,
        start_index=0,
        max_samples=100,  # Set to None to process entire dataset
    )
