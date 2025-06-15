import json
import random

from faker import Faker


def base_output_example():
    # Base example that stays consistent
    return [
        {
            "sentence": "The new smartphone camera produces amazing low-light photos.",
            "labels": random.choices(
                ["technology", "product_review", "positive", "consumer_electronics"],
                k=3,
            ),
            "not_labels": [
                "negative_review",
                "software_issue",
                "customer_complaint",
                "technical_problem",
            ],
        },
        {
            "sentence": "How do I reset my password for the company portal?",
            "labels": random.choices(
                ["question", "technical_support", "workplace", "instruction_request"],
                k=3,
            ),
            "not_labels": [
                "product_review",
                "complaint",
                "announcement",
                "marketing_content",
            ],
        },
        {
            "sentence": "The quarterly financial report shows a significant increase in revenue compared to last year. Our company has successfully expanded into new markets, particularly in the Asia-Pacific region, where we've seen a 35% growth in customer acquisition. The board of directors is optimistic about maintaining this momentum through strategic partnerships and continued investment in research and development.",
            "labels": random.choices(
                [
                    "business",
                    "financial_report",
                    "corporate_communication",
                    "formal",
                    "positive_outlook",
                ],
                k=4,
            ),
            "not_labels": [
                "negative_outlook",
                "technical_documentation",
                "customer_complaint",
                "casual_conversation",
                "product_advertisement",
            ],
        },
    ]


def random_labels():
    fake = Faker()
    # Simple, controlled random elements
    random_domain = random.choices(
        [
            fake.job().lower().replace(" ", "_"),
            fake.job_female().lower().replace(" ", "_"),
            fake.job_male().lower().replace(" ", "_"),
            fake.job().lower().replace(" ", "_"),
            fake.job_male().lower().replace(" ", "_"),
        ],
        k=4,
    )

    random_industry = random.choices(
        [
            "retail",
            "manufacturing",
            "consulting",
            "media",
            "automotive",
            "food_service",
        ],
        k=4,
    )

    random_tone = random.choices(
        ["professional", "casual", "technical", "friendly", "formal", "conversational"],
        k=4,
    )
    return random_domain, random_industry, random_tone


def generate_prompt(num_samples: int, min_labels: int, max_labels: int):
    output_example = base_output_example()

    # Simple, controlled random elements
    random_domain, random_industry, random_tone = random_labels()

    return f"""**You are an expert data generator for machine learning classification tasks.**

**TASK**: Generate **exactly {num_samples}** diverse text examples for zero-shot classification training. Each example must include a **text sample**, a list of **descriptive labels**, and a list of **hard negative labels**. The text and labels will be used to train or evaluate classifiers.

**IMPORTANT**:
There is **no predefined list of labels** or topics. You must create them based on the content of each generated text.
The **examples below are illustrative only** and must not be reused or replicated.

**TEXT LENGTH DIVERSITY**:
Generate a **diverse mix of text lengths** to ensure comprehensive training data: Short sentences, medium sentences, and longer paragraphs.
Aim for approximately **1/3 short, 1/3 medium, 1/3 paragraph** length distribution.

**WHAT IS A LABEL**:
A **label** is a category that describes some aspect of the text. This can relate to its **topic**, **domain**, **intent**, **tone**, **format**, or **style**. Labels help a model understand what the text is about or how it is written.

**WHAT IS A NOT_LABEL (HARD NEGATIVE)**:
A **not_label** is a label that could plausibly apply to similar texts but does NOT apply to this specific text. These should be **challenging negatives** that test the model's ability to distinguish subtle differences. For example:
- For a positive product review: "negative_review", "customer_complaint", "technical_issue"
- For a technical question: "product_advertisement", "corporate_announcement", "social_media_post"
- For formal business communication: "casual_conversation", "personal_story", "entertainment_content"

**HARD NEGATIVE REQUIREMENTS**:
- NOT_LABELS should be **semantically related** but **contextually incorrect**
- They should be **plausible distractors** that could confuse a weak model
- Avoid obvious negatives (e.g., "cooking" for a tech review)
- Focus on **subtle distinctions** (tone, intent, domain nuances)

**LABEL INSPIRATION** (use these as inspiration, but create your own unique labels):

* Content types: "news_article", "product_review", "email", "social_media_post", "instruction"
* Domains: "{'", "'.join(random_domain)}"
* Sentiment/tone: "{'", "'.join(random_tone)}"
* Intent/function: "question", "request", "opinion", "instruction_request", "announcement"
* Industry: "{'", "'.join(random_industry)}"
... and many more

**REQUIREMENTS**:

* Generate **{num_samples}** text entries, each with a unique and realistic text sample
* **Vary text length**: Include short sentences, medium sentences, and longer paragraphs
* To reduce bias, ensure diversity in both the **content** and the **labels**
* Topics should vary widely: include areas like business, science, health, technology, culture, education, etc.
* Vary the **text type**: include statements, instructions, questions, reviews, announcements, complaints, etc.
* Use different **writing styles**: technical, conversational, promotional, formal, casual, etc.
* Assign **{min_labels} to {max_labels}** relevant and informative labels to each entry
* Assign **{min_labels} to {max_labels}** hard negative labels that are plausible but incorrect
* Labels must be tailored to the content; do not repeat generic sets across examples
* **NOT_LABELS must be challenging distractors**, not obvious negatives
* Ensure **maximum diversity** in both the **content**, **text length**, and the **labels**

**OUTPUT FORMAT**:
Return only a **valid JSON array** of size **{num_samples}**, with each object containing:

* "sentence": the generated text (can be a sentence or paragraph)
* "labels": a list of {min_labels}-{max_labels} descriptive strings that DO apply
* "not_labels": a list of {min_labels}-{max_labels} hard negative labels that do NOT apply but could be plausible

**Example (for illustration only)**:

```json
{json.dumps(output_example, indent=2)}
```
"""
