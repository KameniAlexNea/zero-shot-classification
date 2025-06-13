# Zero-Shot Text Classification Dataset Generator

A clean, simple tool for generating high-quality synthetic datasets for zero-shot text classification using Ollama-hosted LLMs.

## Features

- **High-Quality Data Generation**: Uses expert prompting with clear label definitions and examples
- **Intelligent Diversity**: Randomized temperature and seeds for maximum data variety
- **Batch Processing**: Efficiently generate large datasets with automatic batch saving
- **Data Safety**: Each batch saved immediately to prevent data loss
- **Simple Interface**: Clean CLI with minimal, focused functionality

## What Are Labels?

Labels are categories that describe what a text is about, its purpose, tone, domain, or type. Our generator creates diverse labels across:

- **Content Type**: "news_article", "product_review", "social_media_post", "email"
- **Domain**: "technology", "sports", "finance", "healthcare", "entertainment" 
- **Sentiment**: "positive", "negative", "neutral", "complaint", "praise"
- **Intent**: "question", "request", "announcement", "opinion", "fact"
- **Style**: "formal", "casual", "technical", "promotional", "educational"

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

```bash
# Generate 100 samples with default settings
python generic_data.py

# Generate 1000 samples with custom parameters  
python generic_data.py --num-samples 1000 --batch-size 25 --min-labels 2 --max-labels 4

# Use a different Ollama model
python generic_data.py --model mistral --num-samples 500 --output my_dataset.json

# Generate large datasets with batch processing
python generic_data.py --num-samples 10000 --batch-size 50 --batch-dir my_batches
```

### Programmatic Usage

```python
from generic_data import OllamaClient

# Initialize client
client = OllamaClient(model="llama2")

# Generate small dataset
dataset = client.generate_dataset(num_samples=100, min_labels=1, max_labels=5)

# Generate large dataset with batch processing
large_dataset = client.generate_large_dataset(
    total_samples=10000,
    batch_size=50, 
    output_dir="my_batches"
)
```

## Parameters

- `--model`: Ollama model to use (default: "llama2")
- `--num-samples`: Number of samples to generate (default: 100)
- `--batch-size`: Batch size for large datasets (default: 50)
- `--min-labels`: Minimum labels per sentence (default: 1)
- `--max-labels`: Maximum labels per sentence (default: 5)
- `--output`: Output JSON file path (default: "synthetic_data.json")
- `--batch-dir`: Directory for batch files (default: "batches")

## Dataset Format

Generated datasets use this JSON structure with diverse, well-labeled examples:

```json
[
  {
    "sentence": "The new smartphone camera produces amazing low-light photos.",
    "labels": ["technology", "product_review", "positive", "consumer_electronics"]
  },
  {
    "sentence": "How do I reset my password for the company portal?", 
    "labels": ["question", "technical_support", "workplace", "instruction_request"]
  }
]
```

## Data Quality Features

- **Randomized Generation**: Each batch uses different temperature (0.7-0.9) and random seeds
- **Diverse Content**: Covers multiple domains, writing styles, and text types  
- **Smart Labels**: Contextually relevant labels across content type, domain, sentiment, and intent
- **Batch Safety**: Each batch immediately saved to prevent data loss
- **Scalable**: Efficiently handles small datasets to millions of samples

## Requirements

- Python 3.7+
- Ollama server running locally
- Required packages: `ollama`, `llm-output-parser`