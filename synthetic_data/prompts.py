import json
import random
from faker import Faker


def get_subjects(choice=3):
    """Get random subjects from the subjects file."""
    with open("synthetic_data/all_subjects.txt") as f:
        subjects = f.readlines()
    return "\n".join(random.choices(subjects, k=choice))


def create_base_examples():
    """Create consistent base examples for the prompt."""
    return [
        {
            "sentence": "The new smartphone camera produces amazing low-light photos.",
            "labels": ["technology", "product_review", "positive", "consumer_electronics"],
            "not_labels": ["negative_review", "software_issue", "customer_complaint", "technical_problem"],
        },
        {
            "sentence": "How do I reset my password for the company portal?",
            "labels": ["question", "technical_support", "workplace", "instruction_request"],
            "not_labels": ["product_review", "complaint", "announcement", "marketing_content"],
        }
    ]


def get_random_label_categories():
    """Generate random label categories for inspiration."""
    fake = Faker()
    
    domains = [
        fake.job().lower().replace(" ", "_") for _ in range(4)
    ]
    
    industries = random.choices([
        "retail", "manufacturing", "consulting", "media", 
        "automotive", "food_service", "healthcare", "education"
    ], k=4)
    
    tones = random.choices([
        "professional", "casual", "technical", "friendly", 
        "formal", "conversational", "academic", "promotional"
    ], k=4)
    
    return domains, industries, tones


def generate_prompt(num_samples: int, min_labels: int, max_labels: int):
    """
    Generate a comprehensive prompt for creating diverse text classification examples.
    
    Args:
        num_samples: Number of examples to generate
        min_labels: Minimum number of labels per example
        max_labels: Maximum number of labels per example
    
    Returns:
        Formatted prompt string
    """
    base_examples = create_base_examples()
    domains, industries, tones = get_random_label_categories()
    subject_topics = get_subjects(5)
    
    prompt = f"""**TASK: Generate Diverse Text Classification Data**

You are an expert data generator creating **{num_samples}** diverse text examples for zero-shot classification training.

**CORE REQUIREMENTS:**
• Generate exactly {num_samples} unique text samples
• Each sample needs {min_labels}-{max_labels} descriptive labels that DO apply
• Each sample needs {min_labels}-{max_labels} hard negative labels that do NOT apply
• Ensure maximum diversity in content, length, and style

**TEXT DIVERSITY GUIDELINES:**

**Content Variety:**
• Topics: business, science, health, technology, culture, education, etc.
• Text types: statements, questions, reviews, instructions, announcements
• Writing styles: technical, conversational, formal, casual, promotional

**LABEL GUIDELINES:**

**What is a Label:**
A category describing the text's topic, domain, intent, tone, format, or style.

**What is a Hard Negative (not_label):**
A plausible but incorrect label that could confuse a weak model. Focus on subtle distinctions rather than obvious negatives.

**Examples of Good Hard Negatives:**
• For positive review → "negative_review", "customer_complaint"
• For technical question → "product_advertisement", "social_media_post"
• For formal communication → "casual_conversation", "personal_story"

**LABEL INSPIRATION:**
Use these as inspiration but create unique labels:

• Content types: "news_article", "product_review", "email", "social_media_post"
• Domains: "{', '.join(domains)}"
• Tones: "{', '.join(tones)}"
• Industries: "{', '.join(industries)}"
• Intent: "question", "request", "opinion", "instruction", "announcement"

**TOPIC SUGGESTIONS:**
{subject_topics}


**EXAMPLE OUTPUT (for reference only):**
```json
{json.dumps(base_examples, indent=2)}
```

**FINAL REMINDERS:**
• NO predefined label sets - create based on content
• Focus on CHALLENGING hard negatives, not obvious ones
• Ensure MAXIMUM diversity in all aspects
• Return ONLY the JSON array, nothing else"""
    
    return prompt
