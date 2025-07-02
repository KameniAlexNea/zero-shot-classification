import json

from gliznet.tokenizer import GliZNETTokenizer


def main():
    tokenizer = GliZNETTokenizer.from_pretrained("bert-base-uncased")
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
        res = tokenizer.tokenize_example(ex["text"], ex["labels"])
        expected[ex["name"]] = {
            "text": ex["text"],
            "labels": ex["labels"],
            "input_ids": res["input_ids"].tolist(),
            "attention_mask": res["attention_mask"].tolist(),
            "lmask": res["lmask"].tolist(),
        }
    with open("tests/testing_data/expected_tokenizer_outputs.json", "w") as f:
        json.dump(expected, f, indent=2)


if __name__ == "__main__":
    main()
