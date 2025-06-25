import unittest

import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from gliznet.config import LabelName
from gliznet.data import add_tokenized_function, collate_fn


class DummyTokenizer:
    def __init__(self, seq_len: int):
        self.seq_len = seq_len
        # dummy special tokens
        self.pad_token_id = 0

    def __call__(self, text, labels, return_tensors=None, pad=None):
        # Handle both single text and list of texts
        if isinstance(text, list):
            batch_size = len(text)
        else:
            batch_size = 1

        # produce fixed-length outputs with batch dimension
        input_ids = (
            torch.arange(self.seq_len, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        attention_mask = torch.ones(batch_size, self.seq_len, dtype=torch.long)
        lmask = torch.zeros(batch_size, self.seq_len, dtype=torch.bool)

        # mark one label position for test
        if self.seq_len > 1:
            lmask[:, 1] = True

        # If single text, squeeze batch dimension to match expected behavior
        # if batch_size == 1 and not isinstance(text, list):
        #     input_ids = input_ids.squeeze(0)
        #     attention_mask = attention_mask.squeeze(0)
        #     lmask = lmask.squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "lmask": lmask,
        }


class TestDataModule(unittest.TestCase):
    def setUp(self):
        # create a small HuggingFace dataset
        self.hf_data = Dataset.from_dict(
            {
                "text": ["hello", "world"],
                LabelName.ltext: [["a"], ["b", "c"]],
                LabelName.lint: [[1], [0, 1]],
            }
        )
        # use dummy tokenizer that returns seq_len=4
        self.tokenizer = DummyTokenizer(seq_len=4)
        self.dataset = add_tokenized_function(
            hf_dataset=self.hf_data,
            tokenizer=self.tokenizer,
        )

    def test_len(self):
        self.assertEqual(len(self.dataset), 2)

    def test_getitem_shapes(self):
        item = self.dataset[0]
        # after unsqueeze in __getitem__
        self.assertIn("input_ids", item)
        self.assertEqual(len(item), 4)  # input_ids, attention_mask, lmask, labels
        self.assertEqual(item["input_ids"].numel(), 4)
        self.assertEqual(item["attention_mask"].numel(), 4)
        self.assertEqual(item["lmask"].numel(), 4)
        # labels_int was [1] for sample0
        self.assertEqual(item["labels"].shape, (1, 1))

        item2 = self.dataset[1]
        # labels_int was [0,1] for sample1
        self.assertEqual(item2["labels"].shape, (1, 2))

    def test_getitems_shapes(self):
        item = self.dataset[:2]
        # after unsqueeze in __getitem__
        self.assertIn("input_ids", item)
        self.assertEqual(item["input_ids"].shape, (2, 4))
        self.assertEqual(item["attention_mask"].shape, (2, 4))
        self.assertEqual(item["lmask"].shape, (2, 4))
        # labels_int was [1] for sample0
        self.assertIsInstance(item["labels"], list)
        self.assertEqual(len(item["labels"]), 2)
        self.assertEqual(item["labels"][0].shape, (1, 1))
        self.assertEqual(item["labels"][1].shape, (1, 2))

    def test_collate_fn(self):
        item0 = self.dataset[0]
        item1 = self.dataset[1]
        batch = collate_fn([item0, item1])
        # stacked tensors
        self.assertEqual(batch["input_ids"].shape, (2, 4))
        self.assertEqual(batch["attention_mask"].shape, (2, 4))
        self.assertEqual(batch["lmask"].shape, (2, 4))
        # labels is list of tensors without batch dim
        self.assertIsInstance(batch["labels"], list)
        self.assertEqual(batch["labels"][0].shape, (1, 1))
        self.assertEqual(batch["labels"][1].shape, (1, 2))

    def test_dataloader_with_collate(self):
        loader = DataLoader(self.dataset, batch_size=2, collate_fn=collate_fn)
        batch = next(iter(loader))
        self.assertEqual(batch["input_ids"].shape, (2, 4))
        self.assertEqual(batch["attention_mask"].shape, (2, 4))
        self.assertEqual(batch["lmask"].shape, (2, 4))
        # check labels list in batch
        self.assertIsInstance(batch["labels"], list)
        self.assertEqual(batch["labels"][0].item(), 1)
        self.assertEqual(batch["labels"][1].tolist(), [[0, 1]])


if __name__ == "__main__":
    unittest.main()
