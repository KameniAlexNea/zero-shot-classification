"""Shared utilities for synthetic data generation."""

import random
from typing import List, Optional

from litellm import completion
from pydantic import BaseModel


class GeneratedSample(BaseModel):
    """Single generated sample with sentence and labels."""
    sentence: str
    labels: List[str]
    not_labels: List[str]


def generate_prompt(text: str) -> str:
    """Generate prompt for creating zero-shot training data from text.
    
    Args:
        text: Input text to process
    
    Returns:
        Formatted prompt string
    """
    return f"""Developer: Given the following text, generate a comprehensive training sample for zero-shot classification.

INPUT TEXT:
<input>
{text}
</input>

TASK REQUIREMENTS:

SENTENCE:
- Produce ONE paraphrased sentence that encapsulates the main idea and meaning of the input text.
- Provide a distilled summary that preserves the key concepts and significance.
- The sentence must be self-contained and understandable without needing the original context.
- Ensure the core message and intent of the original text are maintained, using an impersonal tone.
- Avoid close paraphrases or light rewordings; carry out substantive paraphrasing.

LABELS:
- Generate 5 to 15 descriptive labels that require nuanced, in-depth understanding for accurate classification.
- Each label must be a distinct, non-empty string; the list must contain only strings.
- Do not choose simple, keyword-based labels that are easily matched by text search.
- Include (where possible): content_type, domain, topic, tone, intent, complexity_level, writing_style, methodology, target_audience.
- Emphasize subtle distinctions and semantic interpretation.
- Example labels: step_by_step_tutorial, theoretical_research, practical_application, beginner_friendly, technical_content.
- No required ordering of labels.

NOT_LABELS (Hard Negatives):
- Provide 5 to 15 negative labels that are plausible yet incorrect for the input text.
- Each not_label must be a non-empty string; lists can include only strings.
- Select hard negatives from related domains or overlapping content types to ensure they are challenging.
- They should require careful differentiation from the true labels.
- Example not_labels: unrelated_domains, different_content_types, mismatched_tone, different_methodologies.
- No required ordering for not_labels.

Special Cases & Error Handling:
- If the input text is ambiguous, extremely short, or empty, return the best possible summary and labels based on available context, or provide an empty string for 'sentence' and empty lists for 'labels' and 'not_labels' if no inferences can be made.
- Do not generate labels or not_labels without sufficient textual context.

Set reasoning_effort = medium to ensure balanced thoroughness. Only return a valid JSON object with this exact structure; do not include markdown, comments, or additional text. Each field must be present:
{{
  "sentence": "<one distilled summary sentence (string)>",
  "labels": ["<label1>", "<label2>", ...],
  "not_labels": ["<not_label1>", "<not_label2>", ...]
}}
All fields must always be included.
"""


def generate_sample(
    text: str,
    model: str = "ollama/qwen2.5:14b",
    max_retries: int = 3,
    api_base: str = "http://localhost:11434",
    temperature_range: tuple[float, float] = (0.7, 0.9),
    max_tokens: int = 2048,
) -> Optional[GeneratedSample]:
    """Generate a synthetic sample from text using LiteLLM with Ollama.
    
    Args:
        text: Input text to process
        model: Ollama model to use (format: "ollama/model_name")
        max_retries: Maximum number of retry attempts
        api_base: Ollama API base URL
        temperature_range: Min and max temperature for generation
        max_tokens: Maximum tokens in response
    
    Returns:
        GeneratedSample object or None if generation fails
    """
    prompt = generate_prompt(text)
    
    for attempt in range(max_retries):
        try:
            response = completion(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You're an expert in generating training data for zero-shot classification models. Always respond with valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=random.uniform(*temperature_range),
                max_tokens=max_tokens,
                api_base=api_base,
                response_format=GeneratedSample,
            )
            
            # Parse using Pydantic model
            response_text = response.choices[0].message.content
            parsed = GeneratedSample.model_validate_json(response_text)
            
            return parsed
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                print("Max retries reached, returning None")
                return None
    
    return None
