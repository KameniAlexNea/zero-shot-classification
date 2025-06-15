import unittest
import torch
from transformers import AutoTokenizer

# Adjust the import path based on your project structure
# This assumes 'gliznet' is a package in the parent directory or in PYTHONPATH
# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gliznet.tokenizer import GliZNETTokenizer


class TestGliZNETTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pretrained_model_name = "bert-base-uncased"
        # Load a dummy tokenizer for expected values, GliZNETTokenizer will load its own
        cls.hf_tokenizer = AutoTokenizer.from_pretrained(cls.pretrained_model_name)
        cls.cls_token_id = cls.hf_tokenizer.cls_token_id
        cls.sep_token_id = cls.hf_tokenizer.sep_token_id
        cls.pad_token_id = cls.hf_tokenizer.pad_token_id

    def setUp(self):
        self.tokenizer = GliZNETTokenizer(
            pretrained_model_name_or_path=self.pretrained_model_name
        )
        # Override max_length for predictable test outputs
        self.tokenizer.max_length = 20

    def test_initialization(self):
        self.assertIsNotNone(self.tokenizer.tokenizer)
        self.assertEqual(self.tokenizer.cls_token_id, self.cls_token_id)
        self.assertEqual(self.tokenizer.sep_token_id, self.sep_token_id)
        self.assertEqual(self.tokenizer.pad_token_id, self.pad_token_id)
        self.assertEqual(self.tokenizer.max_length, 20) # Overridden value

    def test_from_pretrained(self):
        tokenizer_from_pretrained = GliZNETTokenizer.from_pretrained(
            self.pretrained_model_name
        )
        # Reset max_length for this instance for fair comparison if it was changed globally
        tokenizer_from_pretrained.max_length = tokenizer_from_pretrained.tokenizer.model_max_length

        self.assertIsNotNone(tokenizer_from_pretrained.tokenizer)
        self.assertEqual(
            tokenizer_from_pretrained.tokenizer.model_max_length,
            self.hf_tokenizer.model_max_length
        )

    def test_batch_tokenize_special_tokens_issue(self):
        # This test highlights that _batch_tokenize currently includes special tokens
        # from the underlying tokenizer, which can be problematic for _build_sequence.
        text = "hello"
        labels = ["labelA", "labelB"]
        
        # Underlying tokenizer adds special tokens by default
        expected_text_tokens = self.hf_tokenizer.encode(text, add_special_tokens=True)
        expected_label_tokens = [
            self.hf_tokenizer.encode(label, add_special_tokens=True) for label in labels
        ]

        text_tokens, label_tokens_list = self.tokenizer._batch_tokenize(text, labels)
        
        self.assertEqual(text_tokens, expected_text_tokens)
        self.assertListEqual(label_tokens_list, expected_label_tokens)
        # For single text, all_labels is List[str], so tokenizer treats it as a batch of sentences
        # self.tokenizer(all_labels) -> tokenizes each label string in all_labels
        # This part of _batch_tokenize might behave differently based on input types
        # If all_labels is List[str] for single text:
        # tokenizer(["labelA", "labelB"]) -> [[CLS,tokA,SEP], [CLS,tokB,SEP]]
        # The original code's _batch_tokenize for single text input:
        # self.tokenizer(all_labels, ...)["input_ids"]
        # This means it tokenizes the list of labels as a batch.
        expected_labels_tokenized_as_batch = self.hf_tokenizer(labels, add_special_tokens=True)["input_ids"]
        self.assertEqual(label_tokens_list, expected_labels_tokenized_as_batch)

        texts_list = ["hello", "world"]
        all_labels_list = [["labelA"], ["labelB", "labelC"]]
        
        expected_texts_tokens_list = [
            self.hf_tokenizer.encode(t, add_special_tokens=True) for t in texts_list
        ]
        # For batch text input, all_labels is List[List[str]]
        # self.tokenizer(labels, ...) for labels in all_labels
        expected_all_labels_tokens_list = []
        for label_group in all_labels_list:
            expected_all_labels_tokens_list.append(
                self.hf_tokenizer(label_group, add_special_tokens=True)["input_ids"]
            )

        texts_tokens, labels_tokens_outer_list = self.tokenizer._batch_tokenize(texts_list, all_labels_list)
        self.assertEqual(texts_tokens, expected_texts_tokens_list)
        self.assertEqual(labels_tokens_outer_list, expected_all_labels_tokens_list)


    def test_truncate_text_tokens(self):
        # Assuming text_tokens and label_tokens *do not* have extra CLS/SEP for this unit test
        # as _build_sequence expects raw tokens.
        # However, _batch_tokenize *does* add them. This test will use raw tokens
        # to test _truncate_text_tokens's own logic.
        self.tokenizer.max_length = 10
        # label_tokens: [[L1], [L2, L3]] -> flat_count = 3. sep_count = 1.
        # reserve = 2 (CLS, SEP_after_text) + 3 (labels) + 1 (SEP_between_labels) = 6
        # allowed = max_length - reserve = 10 - 6 = 4
        label_tokens = [[1001], [1002, 1003]] # Dummy token IDs
        
        text_tokens_long = [2001, 2002, 2003, 2004, 2005] # len 5
        truncated = self.tokenizer._truncate_text_tokens(text_tokens_long, label_tokens)
        self.assertEqual(len(truncated), 4)
        self.assertEqual(truncated, [2001, 2002, 2003, 2004])

        text_tokens_short = [2001, 2002] # len 2
        truncated = self.tokenizer._truncate_text_tokens(text_tokens_short, label_tokens)
        self.assertEqual(len(truncated), 2)
        self.assertEqual(truncated, [2001, 2002])
        
        # Test with empty labels
        # reserve = 2 (CLS, SEP_after_text) + 0 (labels) + 0 (SEP_between_labels) = 2
        # allowed = 10 - 2 = 8
        truncated_empty_labels = self.tokenizer._truncate_text_tokens(text_tokens_long, [])
        self.assertEqual(len(truncated_empty_labels), 5) # min(5, 8) = 5, not truncated

        # Test with text tokens that would result in negative allowed length if not for max(0, allowed)
        self.tokenizer.max_length = 5
        # reserve = 6 (as above)
        # allowed = 5 - 6 = -1. max(0, -1) = 0.
        truncated_zero_allowed = self.tokenizer._truncate_text_tokens(text_tokens_long, label_tokens)
        self.assertEqual(len(truncated_zero_allowed), 0)


    def test_build_sequence(self):
        # Using raw tokens (no CLS/SEP from initial tokenization) as _build_sequence adds them
        text_tokens = [2001, 2002] # "text"
        label_tokens_list = [[3001], [3002, 3003]] # "L1", "L2a L2b"

        sequence, label_mask = self.tokenizer._build_sequence(text_tokens, label_tokens_list)

        expected_sequence = [
            self.cls_token_id,
            2001, 2002,           # text
            self.sep_token_id,
            3001,                 # L1
            self.sep_token_id,    # SEP between L1 and L2
            3002, 3003,           # L2a L2b
        ]
        self.assertEqual(sequence, expected_sequence)

        # label_mask from _build_sequence is for [text_tokens + label_tokens_flat + seps_between_labels]
        # It does NOT cover the initial CLS or the SEP after text_tokens.
        # For text_tokens [2001, 2002] -> mask [0,0]
        # For label1 [3001] -> mask [1]
        # For SEP between labels -> mask [0]
        # For label2 [3002, 3003] -> mask [1,0]
        expected_label_mask = [0, 0, 1, 0, 1, 0]
        self.assertEqual(label_mask, expected_label_mask)
        self.assertEqual(len(label_mask), len(text_tokens) + sum(len(lab) for lab in label_tokens_list) + len(label_tokens_list) -1 )
        self.assertNotEqual(len(sequence), len(label_mask), "Label mask from _build_sequence is shorter than sequence")

    def test_pad_and_mask(self):
        self.tokenizer.max_length = 10
        # Sequence: [CLS, T1, T2, SEP, L1, SEP, L2] (len 7)
        token_ids = [self.cls_token_id, 1, 2, self.sep_token_id, 10, self.sep_token_id, 20]
        # Flawed label_mask from _build_sequence: [0,0,1,0,1] (len 5, for T1,T2,L1,SEP_btwn,L2)
        # This mask is 2 elements shorter than token_ids (missing CLS, SEP_after_text coverage)
        label_mask_from_build = [0, 0, 1, 0, 1] 

        pad_len = self.tokenizer.max_length - len(token_ids) # 10 - 7 = 3
        
        # Right padding
        result_right = self.tokenizer._pad_and_mask(token_ids, label_mask_from_build)
        expected_input_ids_right = token_ids + [self.pad_token_id] * pad_len
        expected_attn_mask_right = [1] * len(token_ids) + [0] * pad_len
        # label_mask = label_mask_from_build + [False] * pad_len
        # len = len(label_mask_from_build) + pad_len = 5 + 3 = 8
        # This is max_length - 2
        expected_label_mask_right = label_mask_from_build + [False] * pad_len

        self.assertEqual(result_right["input_ids"], expected_input_ids_right)
        self.assertEqual(result_right["attention_mask"], expected_attn_mask_right)
        self.assertEqual(result_right["label_mask"], expected_label_mask_right)
        self.assertEqual(len(result_right["label_mask"]), self.tokenizer.max_length - (len(token_ids) - len(label_mask_from_build)))
        self.assertEqual(len(result_right["label_mask"]), self.tokenizer.max_length - 2)


        # Left padding - simulate by setting padding_side on the mock tokenizer if possible
        # For now, test as if padding_side was 'left'
        original_padding_side = self.tokenizer.tokenizer.padding_side
        self.tokenizer.tokenizer.padding_side = "left"
        result_left = self.tokenizer._pad_and_mask(token_ids, label_mask_from_build)
        self.tokenizer.tokenizer.padding_side = original_padding_side # reset

        expected_input_ids_left = [self.pad_token_id] * pad_len + token_ids
        expected_attn_mask_left = [0] * pad_len + [1] * len(token_ids)
        # label_mask = [False] * pad_len + label_mask_from_build
        # len = pad_len + len(label_mask_from_build) = 3 + 5 = 8
        expected_label_mask_left = [False] * pad_len + label_mask_from_build
        
        self.assertEqual(result_left["input_ids"], expected_input_ids_left)
        self.assertEqual(result_left["attention_mask"], expected_attn_mask_left)
        self.assertEqual(result_left["label_mask"], expected_label_mask_left)
        self.assertEqual(len(result_left["label_mask"]), self.tokenizer.max_length - 2)

    def test_create_label_mask(self):
        # This method seems to generate a more 'correct' label mask for the whole sequence
        # Sequence: [CLS, T1, T2, SEP_text, L1a, L1b, SEP_labels, L2a]
        sequence_ids = [self.cls_token_id, 1, 2, self.sep_token_id, 10, 11, self.sep_token_id, 20]
        # label_tokens are List[List[int]] (token ids)
        label_tokens_ids = [[10, 11], [20]] # Corresponds to L1a,L1b and L2a

        # The method expects List[str] for sequence and List[List[str]] for label_tokens
        # but uses self.sep_token_id (int) for .index(). This is inconsistent.
        # We test its logic assuming sequence is List[int] and label_tokens is List[List[int]]
        # for len() calculations.
        
        mask = self.tokenizer._create_label_mask(sequence_ids, label_tokens_ids)
        
        # Expected: Mask is True at start of L1a (idx 4) and start of L2a (idx 7)
        # [F, F, F, F, T, F, F, T]
        # CLS T1 T2 SEP L1a L1b SEP L2a
        #  0   1  2   3   4   5   6   7
        # sep_token_id is at index 3. start = 3 + 1 = 4.
        # Loop 1 (label [10,11]): mask[4]=True. idx = 4 + len([10,11]) + 1 = 4 + 2 + 1 = 7.
        # Loop 2 (label [20]):    mask[7]=True. idx = 7 + len([20]) + 1 = 7 + 1 + 1 = 9.
        expected_mask = [False, False, False, False, True, False, False, True]
        self.assertEqual(mask, expected_mask)
        self.assertEqual(len(mask), len(sequence_ids))

    def test_tokenize_example_padding_and_tensors(self):
        self.tokenizer.max_length = 25 # Reset for this test
        text = "This is a sample text."
        labels = ["positive", "negative", "neutral"]

        # Tokenize text and labels (raw, without special tokens for _build_sequence input)
        # Note: _batch_tokenize in actual code *adds* special tokens. This test simplifies
        # to focus on tokenize_example's composition if _batch_tokenize was "fixed".
        # To test current behavior, we'd need to use outputs of actual _batch_tokenize.
        
        # Let's use the actual _batch_tokenize behavior to test the full pipeline
        # raw_text_tokens_from_hf = self.hf_tokenizer.encode(text, add_special_tokens=False)
        # raw_label_tokens_from_hf = [self.hf_tokenizer.encode(l, add_special_tokens=False) for l in labels]

        # Simulate what _batch_tokenize would pass (with special tokens)
        # This is the problematic part: text_tokens and label_tokens will have CLS/SEP
        # text_tokens_with_specials = self.hf_tokenizer.encode(text, add_special_tokens=True)
        # _batch_tokenize for single text input tokenizes labels list as a batch:
        # label_tokens_list_with_specials = self.hf_tokenizer(labels, add_special_tokens=True)["input_ids"]


        # To make _truncate_text_tokens and _build_sequence work as originally intended,
        # they should receive tokens *without* CLS/SEP.
        # The current GliZNETTokenizer has an issue here.
        # For this test, we will mock _batch_tokenize to return raw tokens
        # to test the rest of the pipeline more cleanly.
        
        original_batch_tokenize = self.tokenizer._batch_tokenize
        def mock_batch_tokenize_raw(text_input, labels_input):
            if isinstance(text_input, str): # single example
                _text_toks = self.hf_tokenizer.encode(text_input, add_special_tokens=False)
                _label_toks = [self.hf_tokenizer.encode(tok, add_special_tokens=False) for tok in labels_input]
                return _text_toks, _label_toks
            # Batch case (not strictly needed for tokenize_example test)
            _texts_toks = [self.hf_tokenizer.encode(t, add_special_tokens=False) for t in text_input]
            _labels_toks_list = []
            for label_group in labels_input:
                 _labels_toks_list.append([self.hf_tokenizer.encode(tok, add_special_tokens=False) for tok in label_group])
            return _texts_toks, _labels_toks_list

        self.tokenizer._batch_tokenize = mock_batch_tokenize_raw
        
        result = self.tokenizer.tokenize_example(text, labels, return_tensors="pt", pad=True)
        
        self.tokenizer._batch_tokenize = original_batch_tokenize # Restore

        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        self.assertIn("label_mask", result)

        self.assertIsInstance(result["input_ids"], torch.Tensor)
        self.assertIsInstance(result["attention_mask"], torch.Tensor)
        self.assertIsInstance(result["label_mask"], torch.Tensor)

        # self.assertEqual(result["input_ids"].shape[0], 1) # Batch dim
        self.assertEqual(result["input_ids"].shape[0], self.tokenizer.max_length)
        self.assertEqual(result["attention_mask"].shape[0], self.tokenizer.max_length)
        
        # Due to the label_mask construction and padding issues:
        # len(label_mask_from_build) = len(text_tokens_raw) + sum(len(l_raw)) + (len(labels_raw)-1)
        # len(token_ids_sequence) = 1+len(text_tokens_raw)+1 + sum(len(l_raw)) + (len(labels_raw)-1)
        # Difference is 2.
        # So, padded label_mask length will be max_length - 2.
        expected_label_mask_len = self.tokenizer.max_length - 2
        self.assertEqual(result["label_mask"].size(0), expected_label_mask_len)
        self.assertEqual(result["label_mask"].dtype, torch.bool)

        # Test with pad=False
        result_no_pad = self.tokenizer.tokenize_example(text, labels, return_tensors=None, pad=False)
        self.assertLessEqual(len(result_no_pad["input_ids"]), self.tokenizer.max_length)
        # label_mask length should be len(input_ids) - 2 if the logic holds
        self.assertEqual(len(result_no_pad["label_mask"]), len(result_no_pad["input_ids"]) - 2)


    def test_tokenize_batch(self):
        self.tokenizer.max_length = 30
        texts = ["First sentence.", "Second short one."]
        all_labels = [["labA", "labB"], ["labC", "labD", "labE"]]

        # Mock _batch_tokenize to return raw tokens for cleaner downstream testing
        original_batch_tokenize = self.tokenizer._batch_tokenize
        def mock_batch_tokenize_raw_batch(texts_input, labels_batch_input):
            _texts_toks = [self.hf_tokenizer.encode(t, add_special_tokens=False) for t in texts_input]
            _labels_toks_list_of_list = []
            for label_group in labels_batch_input: # label_group is List[str]
                 _labels_toks_list_of_list.append(
                     [self.hf_tokenizer.encode(tok, add_special_tokens=False) for tok in label_group]
                 )
            return _texts_toks, _labels_toks_list_of_list
        self.tokenizer._batch_tokenize = mock_batch_tokenize_raw_batch

        result = self.tokenizer.tokenize_batch(texts, all_labels, return_tensors="pt", pad=True)
        self.tokenizer._batch_tokenize = original_batch_tokenize # Restore

        self.assertEqual(result["input_ids"].shape, (len(texts), self.tokenizer.max_length))
        self.assertEqual(result["attention_mask"].shape, (len(texts), self.tokenizer.max_length))
        # label_mask length will be max_length - 2
        expected_label_mask_len = self.tokenizer.max_length - 2
        self.assertEqual(result["label_mask"].shape, (len(texts), expected_label_mask_len))
        self.assertEqual(result["label_mask"].dtype, torch.bool)

        # Test pad=False (return_tensors='pt' should raise error)
        with self.assertRaises(ValueError):
            self.tokenizer.tokenize_batch(texts, all_labels, return_tensors="pt", pad=False)

        result_no_pad_no_pt = self.tokenizer.tokenize_batch(texts, all_labels, return_tensors=None, pad=False)
        self.assertEqual(len(result_no_pad_no_pt["input_ids"]), len(texts))
        for i in range(len(texts)):
            self.assertLessEqual(len(result_no_pad_no_pt["input_ids"][i]), self.tokenizer.max_length)
            # label_mask length should be len(input_ids[i]) - 2
            self.assertEqual(len(result_no_pad_no_pt["label_mask"][i]), len(result_no_pad_no_pt["input_ids"][i]) - 2)


    def test_call_method(self):
        # Test __call__ with single string (should use tokenize_example)
        text = "A single call."
        labels = ["l1", "l2"]
        # Mock tokenize_example to check if it's called
        original_tokenize_example = self.tokenizer.tokenize_example
        called_example = {"called": False}
        def mock_ex(t, lab, return_tensors, pad):
            called_example["called"] = True
            return original_tokenize_example(t,lab,return_tensors,pad)
        self.tokenizer.tokenize_example = mock_ex
        self.tokenizer(text, labels)
        self.assertTrue(called_example["called"])
        self.tokenizer.tokenize_example = original_tokenize_example

        # Test __call__ with list of strings (should use tokenize_batch)
        texts = ["First call.", "Second call."]
        all_labels = [["lA"], ["lB"]]
        original_tokenize_batch = self.tokenizer.tokenize_batch
        called_batch = {"called": False}
        def mock_batch(t, lab, return_tensors, pad):
            called_batch["called"] = True
            return original_tokenize_batch(t,lab,return_tensors,pad)
        self.tokenizer.tokenize_batch = mock_batch
        self.tokenizer(texts, all_labels)
        self.assertTrue(called_batch["called"])
        self.tokenizer.tokenize_batch = original_tokenize_batch


    def test_decode_sequence(self):
        ids_with_pad = [self.cls_token_id, 7592, 2088, self.sep_token_id, self.pad_token_id, self.pad_token_id]
        decoded_str = self.tokenizer.decode_sequence(ids_with_pad)
        # Expected: "hello world" (bert-base-uncased tokens for 7592, 2088)
        # self.hf_tokenizer.decode([7592, 2088]) -> 'hello world'
        # self.hf_tokenizer.decode([101, 7592, 2088, 102], skip_special_tokens=True) -> 'hello world'
        self.assertEqual(decoded_str.strip(), "hello world") # strip for potential extra spaces from decode

    def test_get_vocab_size(self):
        self.assertEqual(self.tokenizer.get_vocab_size(), self.hf_tokenizer.vocab_size)

if __name__ == "__main__":
    unittest.main()

