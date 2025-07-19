import json
import unittest

from gliznet.tokenizer import GliZNETTokenizer


class TestTokenizerIntegration(unittest.TestCase):
    def setUp(self):
        # Initialize tokenizer and load expected outputs
        self.tokenizer = GliZNETTokenizer.from_pretrained(
            "bert-base-uncased", cls_separator_token=";"
        )
        expected_file = "tests/testing_data/expected_tokenizer_outputs.json"
        with open(expected_file, "r") as f:
            self.expected: dict = json.load(f)

    def test_tokenizer_expected_outputs(self):
        for name, data in self.expected.items():
            with self.subTest(example=name):
                text = data["text"]
                labels = data["labels"]
                result = self.tokenizer.tokenize_example(text, labels)
                # Compare each tensor output to expected lists
                self.assertListEqual(
                    result["input_ids"].tolist(),
                    data["input_ids"],
                    f"Mismatch in input_ids for {name}",
                )
                self.assertListEqual(
                    result["attention_mask"].tolist(),
                    data["attention_mask"],
                    f"Mismatch in attention_mask for {name}",
                )
                self.assertListEqual(
                    result["lmask"].tolist(),
                    data["lmask"],
                    f"Mismatch in lmask for {name}",
                )


if __name__ == "__main__":
    unittest.main()
