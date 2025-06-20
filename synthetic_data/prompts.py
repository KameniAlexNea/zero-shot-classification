import json
import random

from faker import Faker


def get_subjects(choice=3):
    subjects = open("synthetic_data/all_subjects.txt").read().splitlines()
    return "\n".join(random.choices(subjects, k=choice))


def base_output_example():
    # Base example that stays consistent
    return [
        {
            "sentence": "The new smartphone camera produces amazing low-light photos.",
            "labels": random.choices(
                ["technology", "product_review", "positive", "consumer_electronics"],
                k=4,
            ),
            "not_labels": [
                "negative_review",
                "software_issue",
                "customer_complaint",
                "technical_problem",
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
                k=2,
            ),
            "not_labels": random.choices(
                [
                    "negative_outlook",
                    "technical_documentation",
                    "customer_complaint",
                    "casual_conversation",
                    "product_advertisement",
                ],
                k=3,
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


def generate_prompt(
    num_samples: int, min_labels: int, max_labels: int, topics: str = None
) -> str:
    output_example = base_output_example()

    # Simple, controlled random elements
    random_domain, random_industry, random_tone = random_labels()

    if topics is None:
        topics = get_subjects(5)

    return f"""**You are an expert data generator for machine learning classification tasks.**

**TASK**: Generate **exactly {num_samples}** diverse text examples for zero-shot classification training. Each example must include a **text sample**, a list of **descriptive labels**, and a list of **hard negative labels**. The text and labels will be used to train or evaluate classifiers.

**TOPIC REQUIREMENTS**:
You **MUST** create text samples that relate to the following topics. Distribute your examples across these topics to ensure comprehensive coverage:
```
{topics}
```

**Each text sample should clearly relate to one or more of these topics.** Use them as the foundation for your content generation, not just as loose inspiration.

**IMPORTANT**:
There is **no predefined list of labels** or topics beyond those listed above. You must create descriptive labels based on the content of each generated text.
The **examples below are illustrative only** and must not be reused or replicated.

**TEXT LENGTH DIVERSITY**:
Generate a **diverse mix of text lengths** to ensure comprehensive training data: Short sentences, medium sentences, and longer paragraphs.

**WHAT IS A LABEL**:
A **label** is a category that describes some aspect of the text. This can relate to its **topic**, **domain**, **intent**, **tone**, **format**, **style**... Labels help a model understand what the text is about or how it is written.

**LABEL COMPLEXITY REQUIREMENTS**:
**AVOID OBVIOUS LABELS** - Both positive and negative labels should require **deep understanding** of the text content, context, and nuances. Labels should NOT be easily identifiable from surface-level keywords or simple pattern matching.

**Examples of what to AVOID**:
- Obvious keyword-based labels (e.g., "positive" for text containing "good", "great")
- Simple sentiment labels without context (e.g., "happy", "sad")
- Generic topic labels that are immediately apparent (e.g., "food" for a restaurant review)

**Examples of COMPLEX labels that require deeper understanding**:
- "strategic_communication" vs "operational_update"
- "implicit_criticism" vs "constructive_feedback" 
- "expertise_demonstration" vs "knowledge_sharing"
- "market_positioning" vs "competitive_analysis"
- "stakeholder_reassurance" vs "performance_justification"

**WHAT IS A NOT_LABEL (HARD NEGATIVE)**:
A **not_label** is a label that could plausibly apply to similar texts but does NOT apply to this specific text. These should be **challenging negatives** that test the model's ability to distinguish subtle differences and require **deep contextual understanding**.

**HARD NEGATIVE REQUIREMENTS**:
- NOT_LABELS should be **semantically related** but **contextually incorrect**
- They should be **plausible distractors** that could confuse a weak model
- Avoid obvious negatives (e.g., "cooking" for a tech review)
- Focus on **subtle distinctions** (tone, intent, domain nuances, implicit meaning)
- Require **deep text analysis** to distinguish from correct labels
- Should be labels that a **surface-level classifier would incorrectly assign**

**Examples of challenging hard negatives**:
- For analytical business text: "emotional_appeal", "personal_anecdote", "sales_pitch"
- For technical explanation: "marketing_content", "opinion_piece", "troubleshooting_guide"
- For formal announcement: "informal_discussion", "speculative_analysis", "customer_testimonial"

**LABEL INSPIRATION** (use these as inspiration, but create your own unique labels):

* Content types: "news_article", "product_review", "email", "social_media_post", "instruction"
* Domains: "{'", "'.join(random_domain)}"
* Sentiment/tone: "{'", "'.join(random_tone)}"
* Intent/function: "question", "request", "opinion", "instruction_request", "announcement"
* Industry: "{'", "'.join(random_industry)}"
... and many more

**REQUIREMENTS**:

* Generate **{num_samples}** text entries, each relating to the provided topics
* **Follow the topic list above** - ensure your text samples connect to these subjects
* **Vary text length**: Include short sentences, medium sentences, and longer paragraphs
* To reduce bias, ensure diversity in both the **content** and the **labels** while staying within the topic scope
* Vary the **text type**: include statements, instructions, questions, reviews, announcements, complaints, etc.
* Use different **writing styles**: technical, conversational, promotional, formal, casual, etc.
* Assign **{min_labels} to {max_labels}** relevant and informative labels to each entry that require **deep understanding**
* Assign **{min_labels} to {max_labels}** hard negative labels that are plausible but incorrect and require **contextual analysis** to distinguish
* Labels must be tailored to the content; do not repeat generic sets across examples
* **Both positive and negative labels must be complex and nuanced** - avoid obvious classifications
* **not_labels must be challenging distractors** that require deep text understanding to reject
* Ensure **maximum diversity** in both the **content**, **text length**, and the **labels** while adhering to the topic requirements

**OUTPUT FORMAT**:
Return only a **valid JSON array** of size **{num_samples}**, with each object containing:

* "sentence": the generated text (can be a sentence or paragraph) that relates to the provided topics
* "labels": a list of {min_labels}-{max_labels} descriptive strings that DO apply (requiring deep understanding)
* "not_labels": a list of {min_labels}-{max_labels} hard negative labels that do NOT apply but could be plausible (requiring contextual analysis to reject)

**Example (for illustration only)**:

```json
{json.dumps(output_example, indent=2)}
```
"""
