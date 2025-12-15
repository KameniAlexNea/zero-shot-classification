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


def generate_prompt(num_samples: int, topics: str = None) -> str:
    base_output_example()

    # Simple, controlled random elements
    random_domain, random_industry, random_tone = random_labels()

    if topics is None:
        topics = get_subjects(5)

    return f"""Generate **exactly {num_samples}** diverse text samples for zero-shot classification training.

**TOPICS** - Create text samples relating to these topics:
<topics>
{topics}
</topics>

**TEXT REQUIREMENTS**:
- Generate a detailed sentence/paragraph/story with `rich context`
- Avoid short or simple storylines
- Vary text types: statements, instructions, questions, reviews, announcements...
- Use diverse writing styles: technical, conversational, formal, casual...

**LABELS**:
- Create 5-20 descriptive labels per sample that require deep understanding
- Avoid obvious keyword-based labels
- Focus on nuanced aspects: tone, intent, domain, context, style

**NOT_LABELS** (Hard Negatives):
- Create 5-20 challenging negative labels that are plausible but incorrect
- Should be semantically related but contextually wrong
- Require deep analysis to distinguish from correct labels

**LABEL INSPIRATION**:
- Content: news_article, product_review, email, instruction...
- Domains: {"', '".join(random_domain)}...
- Tone: {"', '".join(random_tone)}...
- Industry: {"', '".join(random_industry)}...
- Intent: question, request, opinion, announcement...
...

**OUTPUT**: Return valid JSON array with objects containing:
- "sentence": generated text relating to topics
- "labels": list of applicable descriptive labels
- "not_labels": list of challenging hard negatives
"""
