import unittest

import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from gliznet import LabelName
from gliznet.data import add_tokenized_function, collate_fn
from gliznet.tokenizer import GliZNETTokenizer


class TestDataModule(unittest.TestCase):
    def setUp(self):
        # create a small HuggingFace dataset
        self.hf_data = Dataset.from_dict(
            {
                "text": ["hello", "worlds"],
                LabelName.ltext: [["a"], ["b", "c"]],
                LabelName.lint: [[1], [0, 1]],
            }
        )
        self.tokenizer = GliZNETTokenizer.from_pretrained("bert-base-uncased")
        self.dataset = add_tokenized_function(
            hf_dataset=self.hf_data,
            tokenizer=self.tokenizer,
            shuffle_labels=False,  # Disable shuffling for deterministic tests
        )

    def test_len(self):
        self.assertEqual(len(self.dataset), 2)

    def test_getitem_shapes(self):
        item = self.dataset[0]
        item2 = self.dataset[1]
        # after unsqueeze in __getitem__
        self.assertIn("input_ids", item)
        self.assertEqual(len(item), 4)  # input_ids, attention_mask, lmask, labels
        self.assertEqual(item["input_ids"].shape, (512,))
        self.assertEqual(item["attention_mask"].shape, (512,))
        self.assertEqual(item["lmask"].shape, (512,))
        self.assertEqual(item["labels"].shape, (1,))

        # labels_int was [0,1] for sample1
        self.assertEqual(item2["labels"].shape, (2,))

    def test_getitems_shapes(self):
        item = self.dataset[:2]
        # after unsqueeze in __getitem__
        self.assertIn("input_ids", item)
        self.assertEqual(item["input_ids"].shape, (2, 512))
        self.assertEqual(item["attention_mask"].shape, (2, 512))
        self.assertEqual(item["lmask"].shape, (2, 512))
        # labels_int was [1] for sample0
        self.assertIsInstance(item["labels"], list)
        self.assertEqual(len(item["labels"]), 2)
        self.assertEqual(item["labels"][0].shape, (1,))
        self.assertEqual(item["labels"][1].shape, (2,))

    def test_collate_fn(self):
        item0 = self.dataset[0]
        item1 = self.dataset[1]
        item = collate_fn([item0, item1])
        # stacked tensors
        self.assertIn("input_ids", item)
        self.assertEqual(item["input_ids"].shape, (2, 512))
        self.assertEqual(item["attention_mask"].shape, (2, 512))
        self.assertEqual(item["lmask"].shape, (2, 512))
        # labels are now padded with -100
        self.assertIsInstance(item["labels"], torch.Tensor)
        self.assertEqual(item["labels"].shape, (2, 2))  # batch_size x max_labels
        # Check padding values based on the actual data:
        # Sample 0: labels_int=[1] -> [1.0, -100]
        # Sample 1: labels_int=[0, 1] -> [0.0, 1.0]
        self.assertEqual(item["labels"][0, 0].item(), 1.0)  # first sample, first label
        self.assertEqual(item["labels"][0, 1].item(), -100)  # first sample, padding
        self.assertEqual(item["labels"][1, 0].item(), 0.0)  # second sample, first label
        self.assertEqual(
            item["labels"][1, 1].item(), 1.0
        )  # second sample, second label

    def test_dataloader_with_collate(self):
        loader = DataLoader(self.dataset, batch_size=2, collate_fn=collate_fn)
        item = next(iter(loader))
        self.assertIn("input_ids", item)
        self.assertEqual(item["input_ids"].shape, (2, 512))
        self.assertEqual(item["attention_mask"].shape, (2, 512))
        self.assertEqual(item["lmask"].shape, (2, 512))
        # labels are now padded tensor
        self.assertIsInstance(item["labels"], torch.Tensor)
        self.assertEqual(item["labels"].shape, (2, 2))  # batch_size x max_labels


if __name__ == "__main__":
    unittest.main()
