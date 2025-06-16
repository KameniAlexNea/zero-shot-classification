import unittest
from collections import namedtuple

import torch
import torch.nn as nn

from gliznet.model import GliZNetModel


class DummyEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.config = namedtuple("cfg", ("hidden_size",))(hidden_size)

    def forward(self, input_ids, attention_mask=None, return_dict=True, *args, **kwargs):
        batch, seq_len = input_ids.shape
        # last_hidden_state[b,s,:] = input_ids[b,s] repeated
        last_hidden_state = (
            input_ids.unsqueeze(-1).repeat(1, 1, self.config.hidden_size).float()
        )
        if return_dict:
            return namedtuple("out", ("last_hidden_state",))(last_hidden_state)
        return last_hidden_state


class TestGliZNetModel(unittest.TestCase):
    def setUp(self):
        self.hidden_size = 8
        self.model = GliZNetModel.from_pretrained(
            "bert-base-uncased",
            hidden_size=self.hidden_size,
            similarity_metric="dot",
        )
        # replace encoder and align config + bypass proj
        self.model.backbone = DummyEncoder(self.hidden_size)
        self.model.config.hidden_size = self.hidden_size
        self.model.hidden_size = self.hidden_size
        self.model.proj = nn.Identity()
        # sample inputs: batch=2, seq_len=4
        self.input_ids = torch.tensor([[101, 1012, 1013, 1014], [101, 1016, 1001, 0]])
        self.attn = torch.where(self.input_ids > 0, 1, 0)
        # label_mask: mark pos 2 in sample0, none in sample1
        self.label_mask = torch.tensor(
            [[False, False, True, False], [False, False, False, False]]
        )
        # labels only for sample0: one label target 1.0
        self.labels = [torch.tensor([1.0]), torch.tensor([])]

    def test_compute_similarity_dot(self):
        t = torch.tensor([[1.0, 2.0]])
        lab = torch.tensor([[3.0, 4.0]])
        sim = self.model.compute_similarity(t, lab)  # dot / temp
        # expected = 1*3 + 2*4 = 11
        self.assertTrue(torch.allclose(sim, torch.tensor([[11.0]])))

    def test_forward_without_labels(self):
        out = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attn,
            label_mask=self.label_mask,
            labels=None,
        )
        self.assertIn("logits", out)
        self.assertIn("hidden_states", out)

        loss = out.loss
        self.assertIsNone(loss)
        # sample0 has one label => logits tensor of shape (1,)
        self.assertEqual(out["logits"][0].shape, (1, 1))
        # sample1 no labels => zero-length
        self.assertEqual(out["logits"][1].numel(), 0)

    def test_forward_with_labels_and_loss(self):
        out = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attn,
            label_mask=self.label_mask,
            labels=self.labels,
        )
        self.assertIn("loss", out)
        self.assertIsInstance(out["loss"], torch.Tensor)
        self.assertGreaterEqual(out["loss"].item(), 0.0)

    def test_predict(self):
        results = self.model.predict(
            input_ids=self.input_ids,
            attention_mask=self.attn,
            label_mask=self.label_mask,
        )
        # two samples
        self.assertEqual(len(results), 2)
        for idx, res in enumerate(results):
            self.assertIsInstance(res, list)


if __name__ == "__main__":
    unittest.main()
