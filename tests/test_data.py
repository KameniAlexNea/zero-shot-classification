import unittest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from gliznet.data import GliZNetDataset, collate_fn


class DummyTokenizer:
    def __init__(self, seq_len: int):
        self.seq_len = seq_len
        # dummy special tokens
        self.pad_token_id = 0

    def __call__(self, text, labels, return_tensors=None, pad=None):
        # produce fixed-length outputs
        input_ids = torch.arange(self.seq_len, dtype=torch.long)
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)
        label_mask = torch.zeros(self.seq_len, dtype=torch.bool)
        # mark one label position for test
        if self.seq_len > 1:
            label_mask[1] = True
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_mask": label_mask,
        }


class TestDataModule(unittest.TestCase):
    def setUp(self):
        # create a small HuggingFace dataset
        self.hf_data = Dataset.from_dict(
            {
                "text": ["hello", "world"],
                "labels_text": [["a"], ["b", "c"]],
                "labels_int": [[1], [0, 1]],
            }
        )
        # use dummy tokenizer that returns seq_len=4
        self.tokenizer = DummyTokenizer(seq_len=4)
        self.dataset = GliZNetDataset(
            hf_dataset=self.hf_data,
            tokenizer=self.tokenizer,
        )

    def test_len(self):
        self.assertEqual(len(self.dataset), 2)

    def test_getitem_shapes(self):
        item = self.dataset[0]
        # after unsqueeze in __getitem__
        self.assertIn("input_ids", item)
        self.assertEqual(item["input_ids"].shape, (1, 4))
        self.assertEqual(item["attention_mask"].shape, (1, 4))
        self.assertEqual(item["label_mask"].shape, (1, 4))
        # labels_int was [1] for sample0
        self.assertEqual(item["labels"].shape, (1, 1))

        item2 = self.dataset[1]
        # labels_int was [0,1] for sample1
        self.assertEqual(item2["labels"].shape, (1, 2))

    def test_collate_fn(self):
        item0 = self.dataset[0]
        item1 = self.dataset[1]
        batch = collate_fn([item0, item1])
        # stacked tensors
        self.assertEqual(batch["input_ids"].shape, (2, 4))
        self.assertEqual(batch["attention_mask"].shape, (2, 4))
        self.assertEqual(batch["label_mask"].shape, (2, 4))
        # labels is list of tensors without batch dim
        self.assertIsInstance(batch["labels"], list)
        self.assertEqual(batch["labels"][0].shape, (1,1))
        self.assertEqual(batch["labels"][1].shape, (1,2))

    def test_dataloader_with_collate(self):
        loader = DataLoader(self.dataset, batch_size=2, collate_fn=collate_fn)
        batch = next(iter(loader))
        self.assertEqual(batch["input_ids"].shape, (2, 4))
        self.assertEqual(batch["attention_mask"].shape, (2, 4))
        self.assertEqual(batch["label_mask"].shape, (2, 4))
        # check labels list in batch
        self.assertIsInstance(batch["labels"], list)
        self.assertEqual(batch["labels"][0].item(), 1)
        self.assertEqual(batch["labels"][1].tolist(), [[0, 1]])


if __name__ == "__main__":
    unittest.main()
