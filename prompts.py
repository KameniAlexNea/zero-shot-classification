import json

def generate_prompt(num_samples: int, min_labels: int, max_labels: int):
    output_example = [
  {
    "sentence": "The new smartphone camera produces amazing low-light photos.",
    "labels": ["technology", "product_review", "positive", "consumer_electronics"]
  },
  {
    "sentence": "How do I reset my password for the company portal?",
    "labels": ["question", "technical_support", "workplace", "instruction_request"]
  }
]
    return f"""**You are an expert data generator for machine learning classification tasks.**

**TASK**: Generate **exactly {num_samples}** diverse text examples for zero-shot classification training. Each example must include a **text sample** and a list of **descriptive labels**. The text and labels will be used to train or evaluate classifiers.

**IMPORTANT**:
There is **no predefined list of labels** or topics. You must create them based on the content of each generated text.
The **examples below are illustrative only** and must not be reused or replicated.

**WHAT IS A LABEL**:
A **label** is a category that describes some aspect of the text. This can relate to its **topic**, **domain**, **intent**, **tone**, **format**, or **style**. Labels help a model understand what the text is about or how it is written.

**Example label types (not exhaustive or fixed)**:

* Content type: `"news_article"`, `"product_review"`, `"email"`, `"social_media_post"`, `"instruction"`
* Domain: `"technology"`, `"sports"`, `"finance"`, `"healthcare"`, `"education"`, `"entertainment"`
* Sentiment or tone: `"positive"`, `"negative"`, `"neutral"`, `"sarcastic"`, `"formal"`, `"casual"`
* Intent or function: `"question"`, `"request"`, `"opinion"`, `"instruction_request"`, `"announcement"`

**REQUIREMENTS**:

* Generate **{num_samples}** text entries, each with a unique and realistic sentence
* To reduce bias, ensure diversity in both the **content** and the **labels**
* Topics should vary widely: include areas like business, science, health, technology, culture, education, etc.
* Vary the **text type**: include statements, instructions, questions, reviews, announcements, complaints, etc.
* Use different **writing styles**: technical, conversational, promotional, formal, casual, etc.
* Assign **{min_labels} to {max_labels}** relevant and informative labels to each entry
* Labels must be tailored to the content; do not repeat generic sets across examples
* Ensure **maximum diversity** in both the **content** and the **labels**

**OUTPUT FORMAT**:
Return only a **valid JSON array** of size **{num_samples}**, with each object containing:

* `"sentence"`: the generated text
* `"labels"`: a list of {min_labels}-{max_labels} descriptive strings

**Example (for illustration only)**:

```json
{json.dumps(output_example, indent=2)}
```"""