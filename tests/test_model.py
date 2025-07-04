import unittest
from collections import namedtuple

import torch
import torch.nn as nn

from gliznet.model import GliZNetForSequenceClassification


class DummyEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.config = namedtuple("cfg", ("hidden_size",))(hidden_size)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        return_dict=True,
        output_attentions=False,
        *args,
        **kwargs
    ):
        batch, seq_len = input_ids.shape
        # last_hidden_state[b,s,:] = input_ids[b,s] repeated
        last_hidden_state = (
            input_ids.unsqueeze(-1).repeat(1, 1, self.config.hidden_size).float()
        )

        # Create dummy attention weights
        # Shape: (batch, num_heads, seq_len, seq_len)
        # Using 12 heads as a typical value, can be adjusted if needed
        num_heads = 12
        num_layers = 12
        attentions = None

        if output_attentions:
            # Create dummy attention weights - uniform attention for simplicity
            dummy_attention = torch.ones(batch, num_heads, seq_len, seq_len) / seq_len
            if attention_mask is not None:
                # Apply attention mask
                mask = (
                    attention_mask.unsqueeze(1)
                    .unsqueeze(2)
                    .expand(batch, num_heads, seq_len, seq_len)
                )
                dummy_attention = dummy_attention * mask
                # Renormalize
                dummy_attention = dummy_attention / (
                    dummy_attention.sum(dim=-1, keepdim=True) + 1e-8
                )

            # Return attentions for all layers (same pattern repeated)
            attentions = tuple([dummy_attention for _ in range(num_layers)])

        if return_dict:
            return namedtuple("out", ("last_hidden_state", "attentions"))(
                last_hidden_state, attentions
            )
        return last_hidden_state


class TestGliZNetForSequenceClassification(unittest.TestCase):
    def setUp(self):
        self.hidden_size = 8
        self.model = GliZNetForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            projected_dim=self.hidden_size,
            similarity_metric="dot",
        )
        # replace encoder and align config + bypass proj
        setattr(
            self.model, self.model.base_model_prefix, DummyEncoder(self.hidden_size)
        )
        self.model.config.hidden_size = self.hidden_size
        self.model.hidden_size = self.hidden_size
        self.model.proj = nn.Identity()
        # sample inputs: batch=2, seq_len=4
        self.input_ids = torch.tensor([[101, 1012, 1013, 1014], [101, 1016, 1001, 0]])
        self.attn = torch.where(self.input_ids > 0, 1, 0)
        # lmask: mark pos 2 in sample0, none in sample1
        self.lmask = torch.tensor(
            [[False, False, True, False], [False, False, True, False]]
        )
        # labels only for sample0: one label target 1.0
        self.labels = [torch.tensor([1.0]), torch.tensor([0.0])]

    def _create_model_with_metric(self, similarity_metric):
        """Helper method to create a model with a specific similarity metric"""
        model = GliZNetForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            projected_dim=self.hidden_size,
            similarity_metric=similarity_metric,
        )
        setattr(model, model.base_model_prefix, DummyEncoder(self.hidden_size))
        model.config.hidden_size = self.hidden_size
        model.hidden_size = self.hidden_size
        model.proj = nn.Identity()
        return model

    def test_compute_similarity_dot(self):
        t = torch.tensor([[1.0, 2.0]])
        lab = torch.tensor([[3.0, 4.0]])
        sim = self.model.compute_similarity(t, lab)  # dot / temp
        self.assertTrue(torch.allclose(sim, torch.tensor([[5.5]])))

    # Test similarity metric: dot
    def test_similarity_metric_dot(self):
        """Test dot product similarity metric"""
        model = self._create_model_with_metric("dot")

        # Test configuration
        self.assertEqual(model.config.similarity_metric, "dot")

        # Test compute_similarity method
        t = torch.tensor([[1.0, 2.0, 3.0]])
        lab = torch.tensor([[2.0, 1.0, 1.0]])
        expected = torch.tensor([[2.33333]])
        sim = model.compute_similarity(t, lab)
        self.assertTrue(torch.allclose(sim, expected))

        # Test forward pass
        out = model(
            input_ids=self.input_ids,
            attention_mask=self.attn,
            lmask=self.lmask,
            labels=None,
        )
        self.assertIn("logits", out)
        self.assertEqual(out["logits"].shape, (2, 1))

    # Test similarity metric: bilinear
    def test_similarity_metric_bilinear(self):
        """Test bilinear similarity metric"""
        model = self._create_model_with_metric("bilinear")

        # Test configuration
        self.assertEqual(model.config.similarity_metric, "bilinear")

        # Test that bilinear layer is created
        self.assertTrue(hasattr(model, "classifier"))
        self.assertIsInstance(model.classifier, nn.Bilinear)

        # Test forward pass
        out = model(
            input_ids=self.input_ids,
            attention_mask=self.attn,
            lmask=self.lmask,
            labels=None,
        )
        self.assertIn("logits", out)
        self.assertEqual(out["logits"].shape, (2, 1))

    # Test similarity metric: dot_learning
    def test_similarity_metric_dot_learning(self):
        """Test dot_learning similarity metric"""
        model = self._create_model_with_metric("dot_learning")

        # Test configuration
        self.assertEqual(model.config.similarity_metric, "dot_learning")

        # Test that linear layer is created
        self.assertTrue(hasattr(model, "classifier"))
        self.assertIsInstance(model.classifier, nn.Linear)
        self.assertEqual(model.classifier.out_features, 1)
        # self.assertIsNone(model.classifier.bias)  # Should be bias=False

        # Test forward pass
        out = model(
            input_ids=self.input_ids,
            attention_mask=self.attn,
            lmask=self.lmask,
            labels=None,
        )
        self.assertIn("logits", out)
        self.assertEqual(out["logits"].shape, (2, 1))

    def test_all_similarity_metrics_with_labels(self):
        """Test all similarity metrics with labels and loss computation"""
        metrics = ["dot", "bilinear", "dot_learning"]

        for metric in metrics:
            with self.subTest(similarity_metric=metric):
                model = self._create_model_with_metric(metric)

                out = model(
                    input_ids=self.input_ids,
                    attention_mask=self.attn,
                    lmask=self.lmask,
                    labels=self.labels,
                )

                # Check that loss is computed
                self.assertIn("loss", out)
                self.assertIsInstance(out["loss"], torch.Tensor)
                self.assertGreaterEqual(out["loss"].item(), 0.0)

                # Check logits shape
                self.assertIn("logits", out)
                self.assertEqual(out["logits"].shape, (2, 1))

    def test_all_similarity_metrics_predict(self):
        """Test prediction method for all similarity metrics"""
        metrics = ["dot", "bilinear", "dot_learning"]

        for metric in metrics:
            with self.subTest(similarity_metric=metric):
                model = self._create_model_with_metric(metric)

                results = model.predict(
                    input_ids=self.input_ids,
                    attention_mask=self.attn,
                    lmask=self.lmask,
                )

                # Check results structure
                self.assertEqual(len(results), 2)  # Two samples
                for idx, res in enumerate(results):
                    self.assertIsInstance(res, list)
                    self.assertEqual(len(res), 1)  # One label per sample
                    self.assertIsInstance(res[0], float)
                    self.assertGreaterEqual(res[0], 0.0)
                    self.assertLessEqual(res[0], 1.0)  # Should be sigmoid output

    def test_similarity_metric_consistency(self):
        """Test that similarity computations are consistent within each metric"""
        # Test with fixed inputs to ensure deterministic behavior
        fixed_input_ids = torch.tensor([[101, 1000, 2000, 3000], [101, 4000, 5000, 0]])
        fixed_attn = torch.where(fixed_input_ids > 0, 1, 0)
        fixed_lmask = torch.tensor(
            [[False, False, True, False], [False, False, True, False]]
        )

        metrics = ["dot", "bilinear", "dot_learning"]

        for metric in metrics:
            with self.subTest(similarity_metric=metric):
                model1 = self._create_model_with_metric(metric)
                model2 = self._create_model_with_metric(metric)

                # Copy weights from model1 to model2 to ensure identical parameters
                if hasattr(model1, "classifier"):
                    model2.classifier.load_state_dict(model1.classifier.state_dict())

                # Forward pass on both models
                out1 = model1(
                    input_ids=fixed_input_ids,
                    attention_mask=fixed_attn,
                    lmask=fixed_lmask,
                    labels=None,
                )

                out2 = model2(
                    input_ids=fixed_input_ids,
                    attention_mask=fixed_attn,
                    lmask=fixed_lmask,
                    labels=None,
                )

                # Results should be identical
                self.assertTrue(
                    torch.allclose(out1["logits"], out2["logits"], atol=1e-6)
                )

    def test_invalid_similarity_metric(self):
        """Test that invalid similarity metrics raise appropriate errors"""
        with self.assertRaises(ValueError):
            GliZNetForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                projected_dim=self.hidden_size,
                similarity_metric="invalid_metric",
            )

    # Original tests
    def test_forward_without_labels(self):
        out = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attn,
            lmask=self.lmask,
            labels=None,
        )
        self.assertIn("logits", out)
        # self.assertIn("hidden_states", out)

        loss = out.loss
        self.assertIsNone(loss)
        # sample0 has one label => logits tensor of shape (1,)
        self.assertEqual(out["logits"].shape, (2, 1))

    def test_forward_with_labels_and_loss(self):
        out = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attn,
            lmask=self.lmask,
            labels=self.labels,
        )
        self.assertIn("loss", out)
        self.assertIsInstance(out["loss"], torch.Tensor)
        self.assertGreaterEqual(out["loss"].item(), 0.0)

    def test_predict(self):
        results = self.model.predict(
            input_ids=self.input_ids,
            attention_mask=self.attn,
            lmask=self.lmask,
        )
        # two samples
        self.assertEqual(len(results), 2)
        for idx, res in enumerate(results):
            self.assertIsInstance(res, list)


if __name__ == "__main__":
    unittest.main()
