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


SYSTEM_PROMPT = """You are an expert in generating training data for zero-shot classification models.

Your task is to generate comprehensive training samples for zero-shot classification. For each input text, you must:

SENTENCE:
- Generate text (1 to 5 sentences) that is semantically related to and inspired by the input text.
- Create content that captures similar themes, concepts, or topics but is NOT a direct summary or paraphrase of the input.
- CRITICAL: The generated sentence will be used ALONE for classification without access to the source text. It must be completely self-contained, clear, and rich in contextual information.
- The sentence must contain sufficient semantic and contextual cues that enable accurate classification based solely on its content.
- Include implicit indicators of: domain, tone, intent, methodology, complexity level, target audience, and content type within the generated text itself.
- The generated text should be original content that a reader familiar with the input's domain would recognize as related.
- Vary the perspective, style, or specific focus while maintaining topical relevance and contextual clarity.
- Can include questions, statements, explanations, or discussions related to the input's subject matter.
- Avoid writing like you're citing or summarizing the input text. Write as if this is standalone content that must convey its own context.

LABELS:
- Generate 5 to 15 descriptive labels that require deep contextual understanding and semantic interpretation for accurate classification.
- Each label must be a distinct, non-empty string; the list must contain only strings.
- CRITICAL: Avoid simple, keyword-based labels that can be matched through text search or surface-level pattern matching.
- Labels should be context-dependent and require understanding the underlying meaning, intent, structure, and nuanced characteristics of the content.
- Focus on semantic properties that demand inference: implicit tone, writing methodology, target audience sophistication, conceptual depth, pedagogical approach, argumentation style, epistemological stance...
- Include (where possible): content_type, domain, topic, tone, intent, complexity_level, writing_style, methodology, target_audience, discourse_pattern, cognitive_demand...
- Emphasize subtle distinctions that cannot be determined by keyword presence alone.
- Example labels: step_by_step_tutorial, theoretical_research, practical_application, beginner_friendly, technical_content, socratic_questioning, empirical_evidence_based, interdisciplinary_synthesis.
- No required ordering of labels.

NOT_LABELS (Hard Negatives):
- Provide 5 to 15 negative labels that are highly plausible yet definitively incorrect for the input text.
- Each not_label must be a non-empty string; lists can include only strings.
- Select hard negatives from closely related domains, similar content types, or overlapping semantic spaces to maximize difficulty.
- NOT_LABELS should be contextually challenging and require careful semantic differentiation from the true labels.
- Avoid obviously unrelated labels; instead, choose labels that might apply to superficially similar content but fail upon deeper contextual analysis.
- These should test the model's ability to distinguish subtle contextual differences rather than simple keyword mismatches.
- Example not_labels: alternative_methodologies, adjacent_domains, contrasting_tones, different_audience_levels, opposite_intent_patterns.
- No required ordering for not_labels.

Special Cases & Error Handling:
- If the input text is ambiguous, extremely short, or empty, return the best possible summary and labels based on available context, or provide an empty string for 'sentence' and empty lists for 'labels' and 'not_labels' if no inferences can be made.
- Do not generate labels or not_labels without sufficient textual context.

OUTPUT FORMAT:
Always return a valid JSON object with this exact structure; do not include markdown, comments, or additional text:
{
  "sentence": "<one distilled summary sentence (string)>",
  "labels": ["<label1>", "<label2>", ...],
  "not_labels": ["<not_label1>", "<not_label2>", ...]
}
All fields must always be included."""


def generate_prompt(text: str) -> str:
    """Generate prompt for creating zero-shot training data from text.

    Args:
        text: Input text to process

    Returns:
        Formatted prompt string
    """
    return f"""Generate a training sample for the following text:

{text}"""


def generate_sample(
    text: str,
    model: str = "ollama/qwen2.5:14b",
    max_retries: int = 3,
    api_base: str = "http://localhost:11434",
    temperature_range: tuple[float, float] = (0.7, 0.9),
    max_tokens: int = 2048,
    api_key=None,
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
                        "content": SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=random.uniform(*temperature_range),
                max_tokens=max_tokens,
                api_base=api_base,
                response_format=GeneratedSample,
                api_key=api_key,
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
