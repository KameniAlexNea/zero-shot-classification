import json

import torch

from gliznet.model import GliZNetForSequenceClassification
from gliznet.tokenizer import GliZNETTokenizer


def seed_everything(seed: int = 42):
    """Set random seed for reproducibility."""
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    seed_everything(42)
    # Initialize tokenizer and model (random weights)
    model_name = "microsoft/deberta-v3-small"
    tokenizer = GliZNETTokenizer.from_pretrained(model_name)
    model = GliZNetForSequenceClassification.from_pretrained_with_tokenizer(
        model_name, tokenizer
    )
    model.eval()

    examples = [
        {
            "name": "greeting_question",
            "text": "Good morning! How are you doing today?",
            "labels": ["greeting", "question"],
        },
        {
            "name": "animal_proverb",
            "text": "The quick brown fox jumps over the lazy dog.",
            "labels": ["animal", "action", "proverb"],
        },
        {
            "name": "space_news",
            "text": "NASA launches a new satellite to study Earth's atmosphere.",
            "labels": ["science", "space", "environment"],
        },
        {
            "name": "cooking_instruction",
            "text": "To bake a cake, preheat the oven to 350°F and mix flour, sugar, and eggs.",
            "labels": ["cooking", "instruction", "recipe"],
        },
        {
            "name": "long_repetition",
            "text": " ".join(["inconceivable"] * 50),
            "labels": ["long", "repetition"],
        },
    ]
    expected = {}
    for ex in examples:
        enc = tokenizer.tokenize_example(ex["text"], ex["labels"])
        input_ids = enc["input_ids"].unsqueeze(0)
        attention_mask = enc["attention_mask"].unsqueeze(0)
        lmask = enc["lmask"].unsqueeze(0)
        pred = model.predict(input_ids, attention_mask, lmask)
        expected[ex["name"]] = {
            "text": ex["text"],
            "labels": ex["labels"],
            "pred": pred,
        }
    with open("tests/testing_data/expected_model_predict_outputs.json", "w") as f:
        json.dump(expected, f, indent=2)


if __name__ == "__main__":
    main()
