import json
from faker import Faker
import random


def base_output_example():
    # Base example that stays consistent
    return [
        {
            "sentence": "The new smartphone camera produces amazing low-light photos.",
            "labels": random.choices(
                ["technology", "product_review", "positive", "consumer_electronics"],
                k=3,
            ),
        },
        {
            "sentence": "How do I reset my password for the company portal?",
            "labels": random.choices(
                ["question", "technical_support", "workplace", "instruction_request"],
                k=3,
            ),
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

**TASK**: Generate **exactly {num_samples}** diverse text examples for zero-shot classification training. Each example must include a **text sample** and a list of **descriptive labels**. The text and labels will be used to train or evaluate classifiers.

**IMPORTANT**:
There is **no predefined list of labels** or topics. You must create them based on the content of each generated text.
The **examples below are illustrative only** and must not be reused or replicated.

**TEXT LENGTH DIVERSITY**:
Generate a **diverse mix of text lengths** to ensure comprehensive training data:
* **Short sentences** (5-15 words): Simple statements, questions, or commands
* **Medium sentences** (15-30 words): More descriptive or complex single sentences
* **Paragraphs** (30+ words, multiple sentences): Longer text blocks with multiple ideas or detailed descriptions

Aim for approximately **1/3 short, 1/3 medium, 1/3 paragraph** length distribution.

**WHAT IS A LABEL**:
A **label** is a category that describes some aspect of the text. This can relate to its **topic**, **domain**, **intent**, **tone**, **format**, or **style**. Labels help a model understand what the text is about or how it is written.

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
* Labels must be tailored to the content; do not repeat generic sets across examples
* Ensure **maximum diversity** in both the **content**, **text length**, and the **labels**

**OUTPUT FORMAT**:
Return only a **valid JSON array** of size **{num_samples}**, with each object containing:

* "sentence": the generated text (can be a sentence or paragraph)
* "labels": a list of {min_labels}-{max_labels} descriptive strings

**Example (for illustration only)**:

{json.dumps(output_example, indent=2)}
"""
