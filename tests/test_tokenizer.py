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
        # Create tokenizer with default ";" separator for backward compatibility
        self.tokenizer = GliZNETTokenizer(
            pretrained_model_name_or_path=self.pretrained_model_name,
            min_text_token=1,
            cls_separator_token=";",  # Use default ";" for existing tests
        )
        self.tokenizer.max_length = 20

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
        tokenizer_from_pretrained.max_length = (
            tokenizer_from_pretrained.tokenizer.model_max_length
        )
        self.assertIsNotNone(tokenizer_from_pretrained.tokenizer)
        self.assertEqual(
            tokenizer_from_pretrained.tokenizer.model_max_length,
            self.hf_tokenizer.model_max_length,
        )

    def test_batch_tokenize_special_tokens_issue(self):
        text = "hello"
        labels = ["labelA", "labelB"]
        expected_text_tokens = self.hf_tokenizer.encode(text, add_special_tokens=False)
        expected_label_tokens = [
            self.hf_tokenizer.encode(label, add_special_tokens=False)
            for label in labels
        ]
        text_tokens, label_tokens_list = self.tokenizer._batch_tokenize(text, labels)
        self.assertEqual(text_tokens, expected_text_tokens)
        self.assertListEqual(label_tokens_list, expected_label_tokens)
        expected_labels_tokenized_as_batch = self.hf_tokenizer(
            labels, add_special_tokens=False
        )["input_ids"]
        self.assertEqual(label_tokens_list, expected_labels_tokenized_as_batch)

        texts_list = ["hello", "world"]
        all_labels_list = [["labelA"], ["labelB", "labelC"]]
        expected_texts_tokens_list = [
            self.hf_tokenizer.encode(t, add_special_tokens=False) for t in texts_list
        ]
        expected_all_labels_tokens_list = []
        for label_group in all_labels_list:
            expected_all_labels_tokens_list.append(
                self.hf_tokenizer(label_group, add_special_tokens=False)["input_ids"]
            )
        texts_tokens, labels_tokens_outer_list = self.tokenizer._batch_tokenize(
            texts_list, all_labels_list
        )
        self.assertEqual(texts_tokens, expected_texts_tokens_list)
        self.assertEqual(labels_tokens_outer_list, expected_all_labels_tokens_list)

    def test_truncate_text_tokens(self):
        self.tokenizer.max_length = 10
        label_tokens = [[1001], [1002, 1003]]  # Dummy token IDs

        text_tokens_long = [2001, 2002, 2003, 2004, 2005]  # len 5
        truncated = self.tokenizer._truncate_text_tokens(text_tokens_long, label_tokens)
        self.assertEqual(len(truncated), 4)
        self.assertEqual(truncated, [2001, 2002, 2003, 2004])

        text_tokens_short = [2001, 2002]  # len 2
        truncated = self.tokenizer._truncate_text_tokens(
            text_tokens_short, label_tokens
        )
        self.assertEqual(len(truncated), 2)
        self.assertEqual(truncated, [2001, 2002])

        truncated_empty_labels = self.tokenizer._truncate_text_tokens(
            text_tokens_long, []
        )
        self.assertEqual(len(truncated_empty_labels), 5)  # min(5, 8) = 5, not truncated

        self.tokenizer.max_length = 5
        truncated_zero_allowed = self.tokenizer._truncate_text_tokens(
            text_tokens_long, label_tokens
        )
        self.assertEqual(len(truncated_zero_allowed), 1)  # One forced token

    def test_build_sequence(self):
        text_tokens = [2001, 2002]  # "text"
        label_tokens_list = [[3001], [3002, 3003]]  # "L1", "L2a L2b"

        sequence, lmask = self.tokenizer._build_sequence(text_tokens, label_tokens_list)

        # Current implementation uses ';' as separator (token ID 1025) and no final separator
        expected_sequence = [
            self.cls_token_id,  # 101
            2001,
            2002,  # text
            self.sep_token_id,  # 102
            3001,  # L1 (label group 1)
            self.tokenizer.label_sep_id,  # 1025 (';' separator)
            3002,
            3003,  # L2a L2b (label group 2)
        ]
        self.assertEqual(sequence, expected_sequence)

        # Current implementation uses integer label group IDs (1, 2, 3, etc.)
        expected_label_mask = [0, 0, 0, 0, 1, 0, 2, 2]
        self.assertEqual(lmask, expected_label_mask)
        self.assertEqual(
            len(lmask),
            len(text_tokens)
            + sum(len(lab) for lab in label_tokens_list)
            + len(label_tokens_list)
            + 1,  # CLS + text + SEP + labels + separators between labels (not after last)
        )
        self.assertEqual(
            len(sequence),
            len(lmask),
            "Label mask from _build_sequence is shorter than sequence",
        )

    def test_pad_and_mask(self):
        self.tokenizer.max_length = 10
        token_ids = [
            self.cls_token_id,
            1,
            2,
            self.sep_token_id,
            10,
            self.sep_token_id,
            20,
        ]
        # Current implementation uses integer label masks
        label_mask_from_build = [0, 0, 0, 0, 1, 0, 2]

        pad_len = self.tokenizer.max_length - len(token_ids)  # 10 - 7 = 3

        result_right = self.tokenizer._pad_and_mask(token_ids, label_mask_from_build)
        expected_input_ids_right = token_ids + [self.pad_token_id] * pad_len
        expected_attn_mask_right = [1] * len(token_ids) + [0] * pad_len
        expected_label_mask_right = label_mask_from_build + [0] * pad_len

        self.assertEqual(result_right["input_ids"], expected_input_ids_right)
        self.assertEqual(result_right["attention_mask"], expected_attn_mask_right)
        self.assertEqual(result_right["lmask"], expected_label_mask_right)

    def test_tokenize_example_padding_and_tensors(self):
        self.tokenizer.max_length = 25  # Reset for this test
        text = "This is a sample text."
        labels = ["positive", "negative", "neutral"]

        original_batch_tokenize = self.tokenizer._batch_tokenize

        def mock_batch_tokenize_raw(text_input, labels_input):
            if isinstance(text_input, str):  # single example
                _text_toks = self.hf_tokenizer.encode(
                    text_input, add_special_tokens=False
                )
                _label_toks = [
                    self.hf_tokenizer.encode(tok, add_special_tokens=False)
                    for tok in labels_input
                ]
                return _text_toks, _label_toks
            _texts_toks = [
                self.hf_tokenizer.encode(t, add_special_tokens=False)
                for t in text_input
            ]
            _labels_toks_list = []
            for label_group in labels_input:
                _labels_toks_list.append(
                    [
                        self.hf_tokenizer.encode(tok, add_special_tokens=False)
                        for tok in label_group
                    ]
                )
            return _texts_toks, _labels_toks_list

        self.tokenizer._batch_tokenize = mock_batch_tokenize_raw

        result = self.tokenizer.tokenize_example(
            text,
            labels,
        )

        self.tokenizer._batch_tokenize = original_batch_tokenize  # Restore

        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        self.assertIn("lmask", result)

        self.assertIsInstance(result["input_ids"], torch.Tensor)
        self.assertIsInstance(result["attention_mask"], torch.Tensor)
        self.assertIsInstance(result["lmask"], torch.Tensor)

        self.assertEqual(result["input_ids"].shape[0], self.tokenizer.max_length)
        self.assertEqual(result["attention_mask"].shape[0], self.tokenizer.max_length)

        expected_label_mask_len = self.tokenizer.max_length
        self.assertEqual(result["lmask"].size(0), expected_label_mask_len)
        self.assertEqual(
            result["lmask"].dtype, torch.long
        )  # Changed from torch.bool to torch.long

        result_no_pad = self.tokenizer.tokenize_example(
            text,
            labels,
        )
        self.assertLessEqual(len(result_no_pad["input_ids"]), self.tokenizer.max_length)
        self.assertEqual(len(result_no_pad["lmask"]), len(result_no_pad["input_ids"]))

    def test_tokenize_batch(self):
        self.tokenizer.max_length = 30
        texts = ["First sentence.", "Second short one."]
        all_labels = [["labA", "labB"], ["labC", "labD", "labE"]]

        original_batch_tokenize = self.tokenizer._batch_tokenize

        def mock_batch_tokenize_raw_batch(texts_input, labels_batch_input):
            _texts_toks = [
                self.hf_tokenizer.encode(t, add_special_tokens=False)
                for t in texts_input
            ]
            _labels_toks_list_of_list = []
            for label_group in labels_batch_input:  # label_group is List[str]
                _labels_toks_list_of_list.append(
                    [
                        self.hf_tokenizer.encode(tok, add_special_tokens=False)
                        for tok in label_group
                    ]
                )
            return _texts_toks, _labels_toks_list_of_list

        self.tokenizer._batch_tokenize = mock_batch_tokenize_raw_batch

        result = self.tokenizer.tokenize_batch(
            texts,
            all_labels,
        )
        self.tokenizer._batch_tokenize = original_batch_tokenize  # Restore

        self.assertEqual(
            result["input_ids"].shape, (len(texts), self.tokenizer.max_length)
        )
        self.assertEqual(
            result["attention_mask"].shape, (len(texts), self.tokenizer.max_length)
        )
        expected_label_mask_len = self.tokenizer.max_length
        self.assertEqual(result["lmask"].shape, (len(texts), expected_label_mask_len))
        self.assertEqual(
            result["lmask"].dtype, torch.long
        )  # Changed from torch.bool to torch.long

    def test_call_method(self):
        text = "A single call."
        labels = ["l1", "l2"]
        original_tokenize_example = self.tokenizer.tokenize_example
        called_example = {"called": False}

        def mock_ex(
            t,
            lab,
            tdp=0.0,
        ):
            called_example["called"] = True
            return original_tokenize_example(
                t,
                lab,
            )

        self.tokenizer.tokenize_example = mock_ex
        self.tokenizer(text, labels)
        self.assertTrue(called_example["called"])
        self.tokenizer.tokenize_example = original_tokenize_example

        texts = ["First call.", "Second call."]
        all_labels = [["lA"], ["lB"]]
        original_tokenize_batch = self.tokenizer.tokenize_batch
        called_batch = {"called": False}

        def mock_batch(
            t,
            lab,
            tdp=0.0,
        ):
            called_batch["called"] = True
            return original_tokenize_batch(
                t,
                lab,
            )

        self.tokenizer.tokenize_batch = mock_batch
        self.tokenizer(texts, all_labels)
        self.assertTrue(called_batch["called"])
        self.tokenizer.tokenize_batch = original_tokenize_batch

    def test_decode_sequence(self):
        ids_with_pad = [
            self.cls_token_id,
            7592,
            2088,
            self.sep_token_id,
            self.pad_token_id,
            self.pad_token_id,
        ]
        decoded_str = self.tokenizer.decode_sequence(ids_with_pad)
        self.assertEqual(decoded_str.strip(), "hello world")

    def test_get_vocab_size(self):
        self.assertEqual(self.tokenizer.get_vocab_size(), self.hf_tokenizer.vocab_size)


class TestGliZNETTokenizerCustomTokens(unittest.TestCase):
    """Test suite for custom token functionality ([LAB] tokens, auto-detection, etc.)"""

    @classmethod
    def setUpClass(cls):
        cls.pretrained_model_name = "bert-base-uncased"
        cls.hf_tokenizer = AutoTokenizer.from_pretrained(cls.pretrained_model_name)
        cls.original_vocab_size = cls.hf_tokenizer.vocab_size

    def test_custom_lab_token_initialization(self):
        """Test tokenizer initialization with custom [LAB] token"""
        tokenizer = GliZNETTokenizer(
            pretrained_model_name_or_path=self.pretrained_model_name,
            cls_separator_token="[LAB]",
        )

        # Should have custom token
        self.assertTrue(tokenizer.has_custom_tokens())
        self.assertEqual(tokenizer.cls_separator_token, "[LAB]")
        self.assertEqual(tokenizer.get_vocab_size(), self.original_vocab_size + 1)
        self.assertEqual(tokenizer.get_added_tokens_count(), 1)
        self.assertIn("[LAB]", tokenizer.get_additional_special_tokens())
        self.assertFalse(tokenizer.was_auto_detected())

    def test_default_semicolon_separator(self):
        """Test tokenizer with default ';' separator"""
        tokenizer = GliZNETTokenizer(
            pretrained_model_name_or_path=self.pretrained_model_name,
            cls_separator_token=";",
        )

        # Should not have custom tokens
        self.assertFalse(tokenizer.has_custom_tokens())
        self.assertEqual(tokenizer.cls_separator_token, ";")
        self.assertEqual(tokenizer.get_vocab_size(), self.original_vocab_size)
        self.assertEqual(tokenizer.get_added_tokens_count(), 0)
        self.assertEqual(tokenizer.get_additional_special_tokens(), [])
        self.assertFalse(tokenizer.was_auto_detected())

    def test_custom_token_sequence_building(self):
        """Test sequence building with custom [LAB] token"""
        tokenizer = GliZNETTokenizer(
            pretrained_model_name_or_path=self.pretrained_model_name,
            cls_separator_token="[LAB]",
        )

        text = "Hello world"
        labels = ["positive", "negative"]

        result = tokenizer.tokenize_example(text, labels)

        # Check that [LAB] token is in the sequence
        lab_token_id = tokenizer.label_sep_id
        self.assertIn(lab_token_id, result["input_ids"])

        # Decode and check structure
        decoded = tokenizer.decode_sequence(result["input_ids"].tolist())
        self.assertIn("hello world", decoded.lower())
        self.assertIn("positive", decoded.lower())
        self.assertIn("negative", decoded.lower())

    def test_auto_detection_from_saved_tokenizer(self):
        """Test auto-detection of custom tokens from saved tokenizer"""
        import os
        import tempfile

        # Create tokenizer with [LAB] token
        original_tokenizer = GliZNETTokenizer(
            pretrained_model_name_or_path=self.pretrained_model_name,
            cls_separator_token="[LAB]",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save tokenizer
            save_path = os.path.join(temp_dir, "custom_tokenizer")
            original_tokenizer.tokenizer.save_pretrained(save_path)

            # Load with default separator - should auto-detect
            loaded_tokenizer = GliZNETTokenizer.from_pretrained(
                save_path, cls_separator_token=";"  # Default, but should be overridden
            )

            # Should auto-detect [LAB]
            self.assertTrue(loaded_tokenizer.was_auto_detected())
            self.assertEqual(loaded_tokenizer.cls_separator_token, "[LAB]")
            self.assertTrue(loaded_tokenizer.has_custom_tokens())
            self.assertEqual(
                loaded_tokenizer.get_vocab_size(), self.original_vocab_size + 1
            )

    def test_no_duplicate_token_addition(self):
        """Test that tokens aren't added twice"""
        import os
        import tempfile

        # Create and save tokenizer with [LAB]
        tokenizer1 = GliZNETTokenizer(
            pretrained_model_name_or_path=self.pretrained_model_name,
            cls_separator_token="[LAB]",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "tokenizer")
            tokenizer1.tokenizer.save_pretrained(save_path)

            # Load and explicitly set [LAB] again
            tokenizer2 = GliZNETTokenizer.from_pretrained(
                save_path, cls_separator_token="[LAB]"  # Explicitly set again
            )

            # Should still only have 1 additional token
            self.assertEqual(tokenizer2.get_added_tokens_count(), 1)
            self.assertEqual(tokenizer2.get_vocab_size(), self.original_vocab_size + 1)

    def test_tokenization_consistency(self):
        """Test that tokenization results are consistent between custom and default"""
        text = "This is a test sentence"
        labels = ["label1", "label2", "label3"]

        # Default tokenizer
        tokenizer_default = GliZNETTokenizer(
            pretrained_model_name_or_path=self.pretrained_model_name,
            cls_separator_token=";",
        )
        result_default = tokenizer_default.tokenize_example(text, labels)

        # Custom tokenizer
        tokenizer_custom = GliZNETTokenizer(
            pretrained_model_name_or_path=self.pretrained_model_name,
            cls_separator_token="[LAB]",
        )
        result_custom = tokenizer_custom.tokenize_example(text, labels)

        # Both should have same structure
        self.assertEqual(
            result_default["input_ids"].shape, result_custom["input_ids"].shape
        )
        self.assertEqual(
            result_default["attention_mask"].shape,
            result_custom["attention_mask"].shape,
        )
        self.assertEqual(result_default["lmask"].shape, result_custom["lmask"].shape)

        # Label masks should have same structure (counting non-zero elements)
        default_labels = (result_default["lmask"] > 0).sum()
        custom_labels = (result_custom["lmask"] > 0).sum()
        self.assertEqual(default_labels, custom_labels)

    def test_batch_tokenization_with_custom_tokens(self):
        """Test batch tokenization with custom tokens"""
        tokenizer = GliZNETTokenizer(
            pretrained_model_name_or_path=self.pretrained_model_name,
            cls_separator_token="[LAB]",
        )

        texts = ["First text", "Second text"]
        labels = [["pos", "neg"], ["happy", "sad", "neutral"]]

        result = tokenizer.tokenize_batch(texts, labels)

        # Check shapes
        self.assertEqual(len(result["input_ids"]), 2)
        self.assertEqual(len(result["attention_mask"]), 2)
        self.assertEqual(len(result["lmask"]), 2)

        # Check that [LAB] tokens are present
        lab_token_id = tokenizer.label_sep_id
        for input_ids in result["input_ids"]:
            if isinstance(input_ids, torch.Tensor):
                self.assertIn(lab_token_id, input_ids.tolist())

    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        # Empty labels
        tokenizer = GliZNETTokenizer(
            pretrained_model_name_or_path=self.pretrained_model_name,
            cls_separator_token="[LAB]",
        )

        text = "Test text"
        result = tokenizer.tokenize_example(text, [])

        # Should still work with empty labels
        self.assertIsInstance(result, dict)
        self.assertIn("input_ids", result)

        # Single label
        result_single = tokenizer.tokenize_example(text, ["single"])
        decoded = tokenizer.decode_sequence(result_single["input_ids"].tolist())
        self.assertIn("test text", decoded.lower())
        self.assertIn("single", decoded.lower())

    def test_from_pretrained_class_method(self):
        """Test the from_pretrained class method with custom tokens"""
        tokenizer = GliZNETTokenizer.from_pretrained(
            self.pretrained_model_name, cls_separator_token="[CUSTOM]", min_text_token=5
        )

        self.assertEqual(tokenizer.cls_separator_token, "[CUSTOM]")
        self.assertTrue(tokenizer.has_custom_tokens())
        self.assertEqual(tokenizer.min_text_token, 5)
        self.assertEqual(tokenizer.get_vocab_size(), self.original_vocab_size + 1)


if __name__ == "__main__":
    unittest.main()
