"""Pytest tests for gliznet data module (tokenized dataset + collate_fn)."""
import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from gliznet import LabelName
from gliznet.data import add_tokenized_function, collate_fn
from gliznet.tokenizer import GliZNETTokenizer


@pytest.fixture(scope="module")
def dataset():
    hf_data = Dataset.from_dict(
        {
            "text": ["hello", "worlds"],
            LabelName.ltext: [["a"], ["b", "c"]],
            LabelName.lint: [[1], [0, 1]],
        }
    )
    tokenizer = GliZNETTokenizer.from_pretrained("bert-base-uncased", model_max_length=512)
    return add_tokenized_function(
        hf_dataset=hf_data,
        tokenizer=tokenizer,
        shuffle_labels=False,
    )


def test_len(dataset):
    assert len(dataset) == 2


def test_getitem_shapes(dataset):
    item = dataset[0]
    item2 = dataset[1]
    assert "input_ids" in item
    assert len(item) == 4  # input_ids, attention_mask, lmask, labels
    assert item["input_ids"].shape == (512,)
    assert item["attention_mask"].shape == (512,)
    assert item["lmask"].shape == (512,)
    assert item["labels"].shape == (1,)
    assert item2["labels"].shape == (2,)


def test_getitems_shapes(dataset):
    item = dataset[:2]
    assert "input_ids" in item
    assert item["input_ids"].shape == (2, 512)
    assert item["attention_mask"].shape == (2, 512)
    assert item["lmask"].shape == (2, 512)
    assert isinstance(item["labels"], list)
    assert len(item["labels"]) == 2
    assert item["labels"][0].shape == (1,)
    assert item["labels"][1].shape == (2,)


def test_collate_fn(dataset):
    item = collate_fn([dataset[0], dataset[1]])
    assert "input_ids" in item
    assert item["input_ids"].shape == (2, 512)
    assert item["attention_mask"].shape == (2, 512)
    assert item["lmask"].shape == (2, 512)
    assert isinstance(item["labels"], torch.Tensor)
    assert item["labels"].shape == (2, 2)
    # Sample 0: labels_int=[1] -> [1.0, -100]
    # Sample 1: labels_int=[0, 1] -> [0.0, 1.0]
    assert item["labels"][0, 0].item() == 1.0
    assert item["labels"][0, 1].item() == -100
    assert item["labels"][1, 0].item() == 0.0
    assert item["labels"][1, 1].item() == 1.0


def test_dataloader_with_collate(dataset):
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    item = next(iter(loader))
    assert "input_ids" in item
    assert item["input_ids"].shape == (2, 512)
    assert item["attention_mask"].shape == (2, 512)
    assert item["lmask"].shape == (2, 512)
    assert isinstance(item["labels"], torch.Tensor)
    assert item["labels"].shape == (2, 2)
