import json
import os
import unittest

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


class TestModelPredictIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure reproducibility
        seed_everything(42)
        model_name = "bert-base-uncased"
        cls.tokenizer = GliZNETTokenizer.from_pretrained(model_name)
        cls.model = GliZNetForSequenceClassification.from_pretrained(model_name)
        cls.model.eval()
        tests_dir = os.path.dirname(__file__)
        expected_file = os.path.join(tests_dir, "expected_model_predict_outputs.json")
        with open(expected_file, "r") as f:
            cls.expected = json.load(f)

    def test_predict_expected_outputs(self):
        for name, data in self.expected.items():
            with self.subTest(example=name):
                text = data["text"]
                labels = data["labels"]
                enc = self.tokenizer.tokenize_example(text, labels)
                input_ids = enc["input_ids"].unsqueeze(0)
                attention_mask = enc["attention_mask"].unsqueeze(0)
                lmask = enc["lmask"].unsqueeze(0)
                pred = self.model.predict(input_ids, attention_mask, lmask)
                expected_pred = data["pred"]
                # Compare nested probability lists
                self.assertEqual(
                    len(pred), len(expected_pred), f"Length mismatch for {name}"
                )
                for p_list, e_list in zip(pred, expected_pred):
                    self.assertEqual(
                        len(p_list), len(e_list), f"Num labels mismatch for {name}"
                    )
                    for p_val, e_val in zip(p_list, e_list):
                        self.assertAlmostEqual(
                            p_val,
                            e_val,
                            places=6,
                            msg=f"Value mismatch for {name}: {p_val} vs {e_val}",
                        )


if __name__ == "__main__":
    unittest.main()
