import os
import tempfile
import unittest

import torch
from transformers import AutoTokenizer

from gliznet.tokenizer import GliZNETTokenizer


class TestGliZNETTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pretrained_model_name = "bert-base-uncased"
        cls.hf_tokenizer = AutoTokenizer.from_pretrained(cls.pretrained_model_name)
        cls.cls_token_id = cls.hf_tokenizer.cls_token_id
        cls.sep_token_id = cls.hf_tokenizer.sep_token_id
        cls.pad_token_id = cls.hf_tokenizer.pad_token_id

    def setUp(self):
        self.tokenizer = GliZNETTokenizer(
            pretrained_model_name_or_path=self.pretrained_model_name,
            min_text_tokens=1,
            model_max_length=20,
        )

    def test_initialization(self):
        self.assertIsNotNone(self.tokenizer.tokenizer)
        self.assertEqual(self.tokenizer.cls_token_id, self.cls_token_id)
        self.assertEqual(self.tokenizer.sep_token_id, self.sep_token_id)
        self.assertEqual(self.tokenizer.pad_token_id, self.pad_token_id)
        self.assertEqual(self.tokenizer.max_length, 20)

    def test_from_pretrained(self):
        tokenizer_from_pretrained = GliZNETTokenizer.from_pretrained(
            self.pretrained_model_name
        )
        self.assertIsNotNone(tokenizer_from_pretrained.tokenizer)
        self.assertEqual(
            tokenizer_from_pretrained.tokenizer.model_max_length,
            self.hf_tokenizer.model_max_length,
        )

    def test_padding_behavior(self):
        """Test that sequences are padded to model_max_length."""
        result = self.tokenizer([("hi", ["a"])], return_tensors="pt")
        self.assertEqual(result["input_ids"].shape, (1, 20))
        self.assertEqual(result["attention_mask"].shape, (1, 20))
        self.assertEqual(result["lmask"].shape, (1, 20))
        # Positions after actual content should be padding
        seq_len = int(result["attention_mask"][0].sum().item())
        self.assertLess(seq_len, 20)
        self.assertEqual(
            result["input_ids"][0, seq_len:].tolist(),
            [self.pad_token_id] * (20 - seq_len),
        )
        self.assertEqual(
            result["attention_mask"][0, seq_len:].tolist(),
            [0] * (20 - seq_len),
        )

    def test_call_single_vs_batch(self):
        """Test that tokenize() and __call__() return correct shapes."""
        # Single via tokenize()
        single = self.tokenizer.tokenize("A single call.", ["l1", "l2"])
        self.assertEqual(single["input_ids"].shape, (20,))
        self.assertEqual(single["attention_mask"].shape, (20,))
        self.assertEqual(single["lmask"].shape, (20,))

        # Batch via __call__()
        texts = ["First call.", "Second call."]
        all_labels = [["lA"], ["lB"]]
        batch = self.tokenizer(list(zip(texts, all_labels)), return_tensors="pt")
        self.assertEqual(batch["input_ids"].shape, (2, 20))
        self.assertEqual(batch["attention_mask"].shape, (2, 20))
        self.assertEqual(batch["lmask"].shape, (2, 20))

    def test_decode(self):
        ids_with_pad = [
            self.cls_token_id,
            7592,
            2088,
            self.sep_token_id,
            self.pad_token_id,
            self.pad_token_id,
        ]
        decoded_str = self.tokenizer.decode(ids_with_pad, skip_special_tokens=True)
        self.assertEqual(decoded_str.strip(), "hello world")


class TestGliZNETTokenizerCustomTokens(unittest.TestCase):
    """Test suite for custom token functionality ([LAB] tokens, etc.)"""

    @classmethod
    def setUpClass(cls):
        cls.pretrained_model_name = "bert-base-uncased"
        cls.hf_tokenizer = AutoTokenizer.from_pretrained(cls.pretrained_model_name)
        cls.original_vocab_size = cls.hf_tokenizer.vocab_size

    def test_custom_lab_token_initialization(self):
        """Test tokenizer initialization with custom [LAB] token."""
        tokenizer = GliZNETTokenizer(
            pretrained_model_name_or_path=self.pretrained_model_name,
            lab_token="[LAB]",
        )

        self.assertEqual(tokenizer.lab_token, "[LAB]")
        self.assertEqual(len(tokenizer), self.original_vocab_size + 1)
        self.assertIn("[LAB]", tokenizer.tokenizer.all_special_tokens)

    def test_custom_token_sequence_building(self):
        """Test sequence building with custom [LAB] token."""
        tokenizer = GliZNETTokenizer(
            pretrained_model_name_or_path=self.pretrained_model_name,
            lab_token="[LAB]",
        )

        text = "Hello world"
        labels = ["positive", "negative"]

        result = tokenizer.tokenize(text, labels)

        self.assertIn(tokenizer.lab_token_id, result["input_ids"].tolist())
        decoded = tokenizer.decode(result["input_ids"].tolist(), skip_special_tokens=True)
        self.assertIn("hello world", decoded.lower())
        self.assertIn("positive", decoded.lower())
        self.assertIn("negative", decoded.lower())

    def test_saved_tokenizer_preserves_custom_tokens(self):
        """Test that a saved tokenizer preserves the [LAB] special token."""
        original_tokenizer = GliZNETTokenizer(
            pretrained_model_name_or_path=self.pretrained_model_name,
            lab_token="[LAB]",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "custom_tokenizer")
            original_tokenizer.save_pretrained(save_path)

            loaded_tokenizer = GliZNETTokenizer.from_pretrained(
                save_path, lab_token="[LAB]"
            )

            self.assertEqual(loaded_tokenizer.lab_token, "[LAB]")
            self.assertIn("[LAB]", loaded_tokenizer.tokenizer.all_special_tokens)
            self.assertEqual(len(loaded_tokenizer), self.original_vocab_size + 1)

    def test_no_duplicate_token_addition(self):
        """Test that tokens aren't added twice when loading a saved tokenizer."""
        tokenizer1 = GliZNETTokenizer(
            pretrained_model_name_or_path=self.pretrained_model_name,
            lab_token="[LAB]",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "tokenizer")
            tokenizer1.save_pretrained(save_path)

            tokenizer2 = GliZNETTokenizer.from_pretrained(
                save_path, lab_token="[LAB]"
            )

            self.assertEqual(len(tokenizer2), self.original_vocab_size + 1)

    def test_batch_tokenization(self):
        """Test batch tokenization with custom tokens."""
        tokenizer = GliZNETTokenizer(
            pretrained_model_name_or_path=self.pretrained_model_name,
            lab_token="[LAB]",
        )

        texts = ["First text", "Second text"]
        labels = [["pos", "neg"], ["happy", "sad", "neutral"]]

        result = tokenizer(list(zip(texts, labels)), return_tensors="pt")

        self.assertEqual(result["input_ids"].shape[0], 2)
        self.assertEqual(result["attention_mask"].shape[0], 2)
        self.assertEqual(result["lmask"].shape[0], 2)

        for i in range(2):
            self.assertIn(tokenizer.lab_token_id, result["input_ids"][i].tolist())

    def test_edge_cases(self):
        """Test edge cases."""
        tokenizer = GliZNETTokenizer(
            pretrained_model_name_or_path=self.pretrained_model_name,
            lab_token="[LAB]",
        )

        text = "Test text"

        # Empty labels — use __call__ directly since tokenize() guards against empty text_labels
        result = tokenizer([(text, [])], return_tensors="pt")
        self.assertIsInstance(result, dict)
        self.assertIn("input_ids", result)

        # Single label
        result_single = tokenizer.tokenize(text, ["single"])
        decoded = tokenizer.decode(result_single["input_ids"].tolist(), skip_special_tokens=True)
        self.assertIn("test text", decoded.lower())
        self.assertIn("single", decoded.lower())

    def test_from_pretrained_class_method(self):
        """Test the from_pretrained class method with custom tokens."""
        tokenizer = GliZNETTokenizer.from_pretrained(
            self.pretrained_model_name, lab_token="[CUSTOM]", min_text_tokens=5
        )

        self.assertEqual(tokenizer.lab_token, "[CUSTOM]")
        self.assertEqual(tokenizer.min_text_tokens, 5)
        self.assertEqual(len(tokenizer), self.original_vocab_size + 1)


if __name__ == "__main__":
    unittest.main()
