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
        self.model.aggregator.cls_proj = nn.Identity()
        self.model.aggregator.label_proj = nn.Identity()
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
        setattr(model, model.base_model_prefix, DummyEncoder(self.hidden_size))
        model.config.hidden_size = self.hidden_size
        model.hidden_size = self.hidden_size
        model.aggregator.cls_proj = nn.Identity()
        model.aggregator.label_proj = nn.Identity()
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
        model = self._create_model_with_metric("dot_learning")

        # Test configuration
        self.assertEqual(model.config.similarity_metric, "dot_learning")

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
        fixed_lmask = torch.tensor([[0, 0, 1, 0], [0, 0, 1, 0]])

        metrics = ["dot", "bilinear", "dot_learning"]

        for metric in metrics:
            with self.subTest(similarity_metric=metric):
                model1 = self._create_model_with_metric(metric)
                model2 = self._create_model_with_metric(metric)

                # Copy weights from model1 to model2 to ensure identical parameters
                if (
                    hasattr(model1.aggregator.similarity_head, "classifier")
                    and model1.aggregator.similarity_head.classifier is not None
                ):
                    model2.aggregator.similarity_head.classifier.load_state_dict(
                        model1.aggregator.similarity_head.classifier.state_dict()
                    )

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


class TestGliZNetWithCustomTokens(unittest.TestCase):
    """Test suite for model functionality with custom tokens and embedding resizing"""

    def setUp(self):
        self.hidden_size = 8

    def test_resize_token_embeddings(self):
        """Test token embedding resizing functionality"""
        # Create base model
        model = GliZNetForSequenceClassification.from_pretrained("bert-base-uncased")
        original_vocab_size = model.config.vocab_size

        # Resize embeddings
        new_vocab_size = original_vocab_size + 5
        new_embeddings = model.resize_token_embeddings(new_vocab_size)

        # Check that embeddings were resized
        self.assertEqual(new_embeddings.num_embeddings, new_vocab_size)
        self.assertEqual(model.get_input_embeddings().num_embeddings, new_vocab_size)

        # Config gets updated by transformers' resize_token_embeddings
        self.assertEqual(model.config.vocab_size, new_vocab_size)

    def test_from_pretrained_with_tokenizer_default(self):
        """Test model creation with default tokenizer (no custom tokens)"""
        from gliznet.tokenizer import GliZNETTokenizer

        tokenizer = GliZNETTokenizer.from_pretrained(
            "bert-base-uncased", cls_separator_token=";"
        )

        model = GliZNetForSequenceClassification.from_pretrained_with_tokenizer(
            "bert-base-uncased", tokenizer
        )

        # Should not resize embeddings or set separator pooling
        self.assertFalse(model.config.use_separator_pooling)
        self.assertEqual(model.config.vocab_size, tokenizer.get_vocab_size())

    def test_from_pretrained_with_tokenizer_custom(self):
        """Test model creation with custom tokenizer ([LAB] tokens)"""
        from gliznet.tokenizer import GliZNETTokenizer

        tokenizer = GliZNETTokenizer.from_pretrained(
            "bert-base-uncased", cls_separator_token="[LAB]"
        )

        model = GliZNetForSequenceClassification.from_pretrained_with_tokenizer(
            "bert-base-uncased", tokenizer
        )

        # Should resize embeddings and set separator pooling
        self.assertTrue(model.config.use_separator_pooling)
        self.assertEqual(model.config.vocab_size, tokenizer.get_vocab_size())
        self.assertEqual(
            model.get_input_embeddings().num_embeddings, tokenizer.get_vocab_size()
        )

    def test_compute_batch_logits_method_selection(self):
        """Test that correct computation method is selected based on use_separator_pooling"""
        from gliznet.tokenizer import GliZNETTokenizer

        # Create model with custom tokens
        tokenizer = GliZNETTokenizer.from_pretrained(
            "bert-base-uncased", cls_separator_token="[LAB]"
        )

        model = GliZNetForSequenceClassification.from_pretrained_with_tokenizer(
            "bert-base-uncased", tokenizer, projected_dim=self.hidden_size
        )

        # Replace with dummy encoder for testing
        setattr(model, model.base_model_prefix, DummyEncoder(self.hidden_size))
        model.config.hidden_size = self.hidden_size
        model.aggregator.cls_proj = nn.Identity()
        model.aggregator.label_proj = nn.Identity()

        # Test that use_separator_pooling flag is set
        self.assertTrue(model.config.use_separator_pooling)

        # Create test inputs
        text = "Test text"
        labels = ["positive", "negative"]
        batch = tokenizer.tokenize_example(text, labels)

        # Test forward pass works
        with torch.no_grad():
            outputs = model.forward(
                input_ids=batch["input_ids"].unsqueeze(0),
                attention_mask=batch["attention_mask"].unsqueeze(0),
                lmask=batch["lmask"].unsqueeze(0),
            )

        self.assertIsNotNone(outputs.logits)
        self.assertTrue(outputs.logits.numel() > 0)

    def test_model_config_persistence(self):
        """Test that model config correctly stores resize information"""
        from gliznet.tokenizer import GliZNETTokenizer

        tokenizer = GliZNETTokenizer.from_pretrained(
            "bert-base-uncased", cls_separator_token="[CUSTOM]"
        )

        model = GliZNetForSequenceClassification.from_pretrained_with_tokenizer(
            "bert-base-uncased", tokenizer
        )

        # Check that config has been updated properly
        self.assertTrue(hasattr(model.config, "use_separator_pooling"))
        self.assertTrue(model.config.use_separator_pooling)
        self.assertEqual(model.config.vocab_size, tokenizer.get_vocab_size())

    def test_model_inference_with_custom_tokens(self):
        """Test model inference with custom tokens"""
        from gliznet.tokenizer import GliZNETTokenizer

        tokenizer = GliZNETTokenizer.from_pretrained(
            "bert-base-uncased", cls_separator_token="[LAB]"
        )

        model = GliZNetForSequenceClassification.from_pretrained_with_tokenizer(
            "bert-base-uncased", tokenizer, projected_dim=self.hidden_size
        )

        # Replace with dummy encoder
        setattr(model, model.base_model_prefix, DummyEncoder(self.hidden_size))
        model.config.hidden_size = self.hidden_size
        model.aggregator.cls_proj = nn.Identity()
        model.aggregator.label_proj = nn.Identity()

        # Test prediction
        texts = ["Great movie!", "Terrible film."]
        labels = [["positive", "negative"], ["good", "bad"]]

        batch = tokenizer.tokenize_batch(texts, labels)

        predictions = model.predict(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            lmask=batch["lmask"],
        )

        # Should return predictions for both samples
        self.assertEqual(len(predictions), 2)
        for pred in predictions:
            self.assertIsInstance(pred, list)
            self.assertTrue(len(pred) > 0)

    def test_encode_method(self):
        """Test the encode method with resized embeddings"""
        from gliznet.tokenizer import GliZNETTokenizer

        tokenizer = GliZNETTokenizer.from_pretrained(
            "bert-base-uncased", cls_separator_token="[LAB]"
        )

        model = GliZNetForSequenceClassification.from_pretrained_with_tokenizer(
            "bert-base-uncased", tokenizer, projected_dim=self.hidden_size
        )

        # Replace with dummy encoder
        setattr(model, model.base_model_prefix, DummyEncoder(self.hidden_size))
        model.config.hidden_size = self.hidden_size

        # Test encoding
        text = "Test encoding"
        batch = tokenizer.tokenize_example(text, ["label"])

        encoding = model.encode(
            input_ids=batch["input_ids"].unsqueeze(0),
            attention_mask=batch["attention_mask"].unsqueeze(0),
        )

        # Should return CLS token encoding
        self.assertEqual(encoding.shape, (1, self.hidden_size))

    def test_backward_compatibility(self):
        """Test that models without custom tokens still work"""
        from gliznet.tokenizer import GliZNETTokenizer

        # Default tokenizer (no custom tokens)
        tokenizer = GliZNETTokenizer.from_pretrained(
            "bert-base-uncased", cls_separator_token=";"
        )

        model = GliZNetForSequenceClassification.from_pretrained_with_tokenizer(
            "bert-base-uncased", tokenizer, projected_dim=self.hidden_size
        )

        # Replace with dummy encoder
        setattr(model, model.base_model_prefix, DummyEncoder(self.hidden_size))
        model.config.hidden_size = self.hidden_size
        model.aggregator.cls_proj = nn.Identity()
        model.aggregator.label_proj = nn.Identity()

        # Should not use separator pooling
        self.assertFalse(model.config.use_separator_pooling)

        # Test inference still works
        text = "Test text"
        labels = ["positive", "negative"]
        batch = tokenizer.tokenize_example(text, labels)

        with torch.no_grad():
            outputs = model.forward(
                input_ids=batch["input_ids"].unsqueeze(0),
                attention_mask=batch["attention_mask"].unsqueeze(0),
                lmask=batch["lmask"].unsqueeze(0),
            )

        self.assertIsNotNone(outputs.logits)

    def test_similarity_metrics_with_custom_tokens(self):
        """Test different similarity metrics with custom tokens"""
        from gliznet.tokenizer import GliZNETTokenizer

        tokenizer = GliZNETTokenizer.from_pretrained(
            "bert-base-uncased", cls_separator_token="[LAB]"
        )

        metrics = ["dot", "bilinear", "dot_learning"]

        for metric in metrics:
            with self.subTest(metric=metric):
                model = GliZNetForSequenceClassification.from_pretrained_with_tokenizer(
                    "bert-base-uncased",
                    tokenizer,
                    projected_dim=self.hidden_size,
                    similarity_metric=metric,
                )

                # Replace with dummy encoder
                setattr(model, model.base_model_prefix, DummyEncoder(self.hidden_size))
                model.config.hidden_size = self.hidden_size
                model.aggregator.cls_proj = nn.Identity()
                model.aggregator.label_proj = nn.Identity()

                # Test forward pass
                text = "Test text"
                labels = ["positive", "negative"]
                batch = tokenizer.tokenize_example(text, labels)

                with torch.no_grad():
                    outputs = model.forward(
                        input_ids=batch["input_ids"].unsqueeze(0),
                        attention_mask=batch["attention_mask"].unsqueeze(0),
                        lmask=batch["lmask"].unsqueeze(0),
                    )

                self.assertIsNotNone(outputs.logits)
                self.assertEqual(model.config.similarity_metric, metric)


if __name__ == "__main__":
    unittest.main()
