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
            pretrained_model_name_or_path=self.pretrained_model_name, min_text_token=1
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
        expected_text_tokens = self.hf_tokenizer.encode(text, add_special_tokens=True)
        expected_label_tokens = [
            self.hf_tokenizer.encode(label, add_special_tokens=True) for label in labels
        ]
        text_tokens, label_tokens_list = self.tokenizer._batch_tokenize(text, labels)
        self.assertEqual(text_tokens, expected_text_tokens)
        self.assertListEqual(label_tokens_list, expected_label_tokens)
        expected_labels_tokenized_as_batch = self.hf_tokenizer(
            labels, add_special_tokens=True
        )["input_ids"]
        self.assertEqual(label_tokens_list, expected_labels_tokenized_as_batch)

        texts_list = ["hello", "world"]
        all_labels_list = [["labelA"], ["labelB", "labelC"]]
        expected_texts_tokens_list = [
            self.hf_tokenizer.encode(t, add_special_tokens=True) for t in texts_list
        ]
        expected_all_labels_tokens_list = []
        for label_group in all_labels_list:
            expected_all_labels_tokens_list.append(
                self.hf_tokenizer(label_group, add_special_tokens=True)["input_ids"]
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

        sequence, label_mask = self.tokenizer._build_sequence(
            text_tokens, label_tokens_list
        )

        expected_sequence = [
            self.cls_token_id,
            2001,
            2002,  # text
            self.sep_token_id,
            3001,  # L1
            self.sep_token_id,  # SEP between L1 and L2
            3002,
            3003,  # L2a L2b
        ]
        self.assertEqual(sequence, expected_sequence)

        expected_label_mask = [0, 0, 0, 0, 1, 0, 1, 0]
        self.assertEqual(label_mask, expected_label_mask)
        self.assertEqual(
            len(label_mask),
            len(text_tokens)
            + sum(len(lab) for lab in label_tokens_list)
            + len(label_tokens_list)
            - 1
            + 2,
        )
        self.assertEqual(
            len(sequence),
            len(label_mask),
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
        label_mask_from_build = [0, 0, 1, 0, 1]

        pad_len = self.tokenizer.max_length - len(token_ids)  # 10 - 7 = 3

        result_right = self.tokenizer._pad_and_mask(token_ids, label_mask_from_build)
        expected_input_ids_right = token_ids + [self.pad_token_id] * pad_len
        expected_attn_mask_right = [1] * len(token_ids) + [0] * pad_len
        expected_label_mask_right = label_mask_from_build + [False] * pad_len

        self.assertEqual(result_right["input_ids"], expected_input_ids_right)
        self.assertEqual(result_right["attention_mask"], expected_attn_mask_right)
        self.assertEqual(result_right["label_mask"], expected_label_mask_right)
        self.assertEqual(
            len(result_right["label_mask"]),
            self.tokenizer.max_length - (len(token_ids) - len(label_mask_from_build)),
        )
        self.assertEqual(len(result_right["label_mask"]), self.tokenizer.max_length - 2)

        original_padding_side = self.tokenizer.tokenizer.padding_side
        self.tokenizer.tokenizer.padding_side = "left"
        result_left = self.tokenizer._pad_and_mask(token_ids, label_mask_from_build)
        self.tokenizer.tokenizer.padding_side = original_padding_side  # reset

        expected_input_ids_left = [self.pad_token_id] * pad_len + token_ids
        expected_attn_mask_left = [0] * pad_len + [1] * len(token_ids)
        expected_label_mask_left = [False] * pad_len + label_mask_from_build

        self.assertEqual(result_left["input_ids"], expected_input_ids_left)
        self.assertEqual(result_left["attention_mask"], expected_attn_mask_left)
        self.assertEqual(result_left["label_mask"], expected_label_mask_left)
        self.assertEqual(len(result_left["label_mask"]), self.tokenizer.max_length - 2)

    def test_create_label_mask(self):
        sequence_ids = [
            self.cls_token_id,
            1,
            2,
            self.sep_token_id,
            10,
            11,
            self.sep_token_id,
            20,
        ]
        label_tokens_ids = [[10, 11], [20]]  # Corresponds to L1a,L1b and L2a

        mask = self.tokenizer._create_label_mask(sequence_ids, label_tokens_ids)

        expected_mask = [False, False, False, False, True, False, False, True]
        self.assertEqual(mask, expected_mask)
        self.assertEqual(len(mask), len(sequence_ids))

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
            text, labels, return_tensors="pt", pad=True
        )

        self.tokenizer._batch_tokenize = original_batch_tokenize  # Restore

        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        self.assertIn("label_mask", result)

        self.assertIsInstance(result["input_ids"], torch.Tensor)
        self.assertIsInstance(result["attention_mask"], torch.Tensor)
        self.assertIsInstance(result["label_mask"], torch.Tensor)

        self.assertEqual(result["input_ids"].shape[0], self.tokenizer.max_length)
        self.assertEqual(result["attention_mask"].shape[0], self.tokenizer.max_length)

        expected_label_mask_len = self.tokenizer.max_length
        self.assertEqual(result["label_mask"].size(0), expected_label_mask_len)
        self.assertEqual(result["label_mask"].dtype, torch.bool)

        result_no_pad = self.tokenizer.tokenize_example(
            text, labels, return_tensors=None, pad=False
        )
        self.assertLessEqual(len(result_no_pad["input_ids"]), self.tokenizer.max_length)
        self.assertEqual(
            len(result_no_pad["label_mask"]), len(result_no_pad["input_ids"])
        )

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
            texts, all_labels, return_tensors="pt", pad=True
        )
        self.tokenizer._batch_tokenize = original_batch_tokenize  # Restore

        self.assertEqual(
            result["input_ids"].shape, (len(texts), self.tokenizer.max_length)
        )
        self.assertEqual(
            result["attention_mask"].shape, (len(texts), self.tokenizer.max_length)
        )
        expected_label_mask_len = self.tokenizer.max_length
        self.assertEqual(
            result["label_mask"].shape, (len(texts), expected_label_mask_len)
        )
        self.assertEqual(result["label_mask"].dtype, torch.bool)

        with self.assertRaises(ValueError):
            self.tokenizer.tokenize_batch(
                texts, all_labels, return_tensors="pt", pad=False
            )

        result_no_pad_no_pt = self.tokenizer.tokenize_batch(
            texts, all_labels, return_tensors=None, pad=False
        )
        self.assertEqual(len(result_no_pad_no_pt["input_ids"]), len(texts))
        for i in range(len(texts)):
            self.assertLessEqual(
                len(result_no_pad_no_pt["input_ids"][i]), self.tokenizer.max_length
            )
            self.assertEqual(
                len(result_no_pad_no_pt["label_mask"][i]),
                len(result_no_pad_no_pt["input_ids"][i]),
            )

    def test_call_method(self):
        text = "A single call."
        labels = ["l1", "l2"]
        original_tokenize_example = self.tokenizer.tokenize_example
        called_example = {"called": False}

        def mock_ex(t, lab, return_tensors, pad):
            called_example["called"] = True
            return original_tokenize_example(t, lab, return_tensors, pad)

        self.tokenizer.tokenize_example = mock_ex
        self.tokenizer(text, labels)
        self.assertTrue(called_example["called"])
        self.tokenizer.tokenize_example = original_tokenize_example

        texts = ["First call.", "Second call."]
        all_labels = [["lA"], ["lB"]]
        original_tokenize_batch = self.tokenizer.tokenize_batch
        called_batch = {"called": False}

        def mock_batch(t, lab, return_tensors, pad):
            called_batch["called"] = True
            return original_tokenize_batch(t, lab, return_tensors, pad)

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


if __name__ == "__main__":
    unittest.main()
