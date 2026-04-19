import unittest
from collections import namedtuple

import torch
import torch.nn as nn
from transformers import AutoModel

from gliznet.model import GliZNetConfig, GliZNetForSequenceClassification


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
        # last_hidden_state[b,s,:] = input_ids[b,s] repeated, scaled to avoid overflow
        last_hidden_state = (
            input_ids.unsqueeze(-1).repeat(1, 1, self.config.hidden_size).float() / 1000.0
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
        self.model.backbone = DummyEncoder(self.hidden_size)
        self.model.config.hidden_size = self.hidden_size
        self.model.hidden_size = self.hidden_size
        self.model.aggregator.text_projector = nn.Identity()
        self.model.aggregator.label_projector = nn.Identity()
        # sample inputs: batch=2, seq_len=4
        self.input_ids = torch.tensor([[101, 1012, 1013, 1014], [101, 1016, 1001, 0]])
        self.attn = torch.where(self.input_ids > 0, 1, 0)
        # lmask: mark pos 2 as label group 1 in both samples
        self.lmask = torch.tensor([[0, 0, 1, 0], [0, 0, 1, 0]])
        # labels: padded tensor with -100 for padding
        self.labels = torch.tensor([[1.0, -100], [0.0, -100]])

    def _create_model_with_metric(self, similarity_metric):
        """Helper method to create a model with a specific similarity metric"""
        model = GliZNetForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            projected_dim=self.hidden_size,
            similarity_metric=similarity_metric,
        )
        model.backbone = DummyEncoder(self.hidden_size)
        model.config.hidden_size = self.hidden_size
        model.hidden_size = self.hidden_size
        model.aggregator.text_projector = nn.Identity()
        model.aggregator.label_projector = nn.Identity()
        return model

    # Test similarity metric: dot
    def test_similarity_metric_dot(self):
        """Test dot product similarity metric"""
        model = self._create_model_with_metric("dot")

        # Test configuration
        self.assertEqual(model.config.similarity_metric, "dot")

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
        self.assertTrue(hasattr(model.aggregator.similarity_head, "classifier"))
        self.assertIsInstance(model.aggregator.similarity_head.classifier, nn.Bilinear)

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
        model = self._create_model_with_metric("dot")

        # Test configuration
        self.assertEqual(model.config.similarity_metric, "dot")

        # Test that linear layer is created
        self.assertTrue(hasattr(model.aggregator.similarity_head, "classifier"))
        self.assertIsInstance(model.aggregator.similarity_head.classifier, nn.Linear)
        self.assertEqual(model.aggregator.similarity_head.classifier.out_features, 1)

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
        metrics = ["dot", "bilinear", "cosine"]

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
        """Test sigmoid scores from forward pass for all similarity metrics."""
        metrics = ["dot", "bilinear", "cosine"]

        for metric in metrics:
            with self.subTest(similarity_metric=metric):
                model = self._create_model_with_metric(metric)

                out = model(
                    input_ids=self.input_ids,
                    attention_mask=self.attn,
                    lmask=self.lmask,
                    labels=None,
                )

                self.assertEqual(out["logits"].shape, (2, 1))
                self.assertTrue(out["logits"].isfinite().all())

    def test_similarity_metric_consistency(self):
        """Test that similarity computations are consistent within each metric"""
        # Test with fixed inputs to ensure deterministic behavior
        fixed_input_ids = torch.tensor([[101, 1000, 2000, 3000], [101, 4000, 5000, 0]])
        fixed_attn = torch.where(fixed_input_ids > 0, 1, 0)
        fixed_lmask = torch.tensor([[0, 0, 1, 0], [0, 0, 1, 0]])

        metrics = ["dot", "bilinear", "cosine"]

        for metric in metrics:
            with self.subTest(similarity_metric=metric):
                model1 = self._create_model_with_metric(metric)
                model2 = self._create_model_with_metric(metric)

                # Copy all weights to ensure identical parameters
                model2.load_state_dict(model1.state_dict())

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
        out = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attn,
            lmask=self.lmask,
            labels=None,
        )
        self.assertEqual(out["logits"].shape, (2, 1))
        self.assertTrue(out["logits"].isfinite().all())


class TestGliZNetWithCustomTokens(unittest.TestCase):
    """Test suite for model functionality with custom tokens and embedding resizing."""

    def setUp(self):
        from gliznet.tokenizer import GliZNETTokenizer
        self.hidden_size = 8
        self.tokenizer = GliZNETTokenizer.from_pretrained(
            "bert-base-uncased", lab_token="[LAB]"
        )

    def _make_model(self, **kwargs):
        """Create a model and resize embeddings to match the tokenizer."""
        model = GliZNetForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            projected_dim=self.hidden_size,
            **kwargs,
        )
        model.resize_token_embeddings(len(self.tokenizer))
        return model

    def _swap_encoder(self, model):
        model.backbone = DummyEncoder(self.hidden_size)
        model.config.hidden_size = self.hidden_size
        model.aggregator.text_projector = nn.Identity()
        model.aggregator.label_projector = nn.Identity()
        return model

    def test_resize_token_embeddings(self):
        """Test token embedding resizing."""
        model = GliZNetForSequenceClassification.from_pretrained("bert-base-uncased")
        original_vocab_size = model.config.backbone_config.vocab_size
        new_vocab_size = original_vocab_size + 5
        model.resize_token_embeddings(new_vocab_size)
        self.assertEqual(model.config.backbone_config.vocab_size, new_vocab_size)
        self.assertEqual(
            model.backbone.get_input_embeddings().num_embeddings, new_vocab_size
        )

    def test_use_lab_token_flag_default(self):
        """By default use_lab_token_for_labels is False."""
        model = GliZNetForSequenceClassification.from_pretrained("bert-base-uncased")
        self.assertFalse(model.config.use_lab_token_for_labels)

    def test_use_lab_token_flag_custom(self):
        """When use_lab_token_for_labels=True the flag and lab_token_id are stored in config."""
        model = GliZNetForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            use_lab_token_for_labels=True,
            lab_token_id=self.tokenizer.lab_token_id,
        )
        self.assertTrue(model.config.use_lab_token_for_labels)
        self.assertEqual(model.config.lab_token_id, self.tokenizer.lab_token_id)

    def test_forward_with_lab_token_mode(self):
        """Forward pass works when use_lab_token_for_labels=True."""
        model = self._make_model(
            use_lab_token_for_labels=True,
            lab_token_id=self.tokenizer.lab_token_id,
        )
        model = self._swap_encoder(model)

        text = "Test text"
        labels = ["positive", "negative"]
        batch = self.tokenizer.tokenize(text, labels)

        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"].unsqueeze(0),
                attention_mask=batch["attention_mask"].unsqueeze(0),
                lmask=batch["lmask"].unsqueeze(0),
            )

        self.assertIsNotNone(outputs.logits)
        self.assertGreater(outputs.logits.numel(), 0)

    def test_model_inference_batch(self):
        """Batch inference produces one score per (sample, label) pair."""
        model = self._make_model(
            use_lab_token_for_labels=False,
            lab_token_id=self.tokenizer.lab_token_id,
        )
        model = self._swap_encoder(model)

        texts = ["Great movie!", "Terrible film."]
        labels = [["positive", "negative"], ["good", "bad"]]
        batch = self.tokenizer(list(zip(texts, labels)), return_tensors="pt")

        with torch.no_grad():
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                lmask=batch["lmask"],
            )

        scores = torch.sigmoid(out["logits"])
        self.assertGreater(scores.numel(), 0)
        self.assertTrue((scores >= 0.0).all())
        self.assertTrue((scores <= 1.0).all())

    def test_similarity_metrics_with_custom_tokens(self):
        """Different similarity metrics work with resized embeddings."""
        for metric in ["dot", "bilinear", "cosine"]:
            with self.subTest(metric=metric):
                model = self._make_model(similarity_metric=metric)
                model = self._swap_encoder(model)

                batch = self.tokenizer.tokenize("Test text", ["positive", "negative"])

                with torch.no_grad():
                    outputs = model(
                        input_ids=batch["input_ids"].unsqueeze(0),
                        attention_mask=batch["attention_mask"].unsqueeze(0),
                        lmask=batch["lmask"].unsqueeze(0),
                    )

                self.assertIsNotNone(outputs.logits)
                self.assertEqual(model.config.similarity_metric, metric)


class TestBackboneWeightIntegrity(unittest.TestCase):
    """Verify that the backbone inside GliZNet produces identical outputs
    to a standalone AutoModel loaded from the same checkpoint."""

    MODEL_NAME = "bert-base-uncased"

    @classmethod
    def setUpClass(cls):
        from gliznet.tokenizer import GliZNETTokenizer
        tokenizer = GliZNETTokenizer.from_pretrained(cls.MODEL_NAME)
        config = GliZNetConfig(backbone_model=cls.MODEL_NAME)
        cls.gliznet = GliZNetForSequenceClassification.from_backbone_pretrained(
            config, tokenizer=tokenizer
        )
        cls.gliznet.eval()

        cls.automodel = AutoModel.from_pretrained(cls.MODEL_NAME)
        cls.automodel.eval()

        # Simple two-token input
        cls.input_ids = torch.tensor([[101, 7592, 102]])       # [CLS] hello [SEP]
        cls.attention_mask = torch.ones_like(cls.input_ids)

    def test_backbone_outputs_match_automodel(self):
        with torch.no_grad():
            gliznet_out = self.gliznet.backbone(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
                return_dict=True,
            )
            auto_out = self.automodel(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
                return_dict=True,
            )

        self.assertTrue(
            torch.allclose(
                gliznet_out.last_hidden_state,
                auto_out.last_hidden_state,
                atol=1e-5,
            ),
            "GliZNet backbone hidden states differ from standalone AutoModel — "
            "backbone weights were not loaded correctly.",
        )


if __name__ == "__main__":
    unittest.main()
