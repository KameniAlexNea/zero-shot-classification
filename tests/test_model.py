"""
Pytest tests for gliznet model: forward pass, similarity metrics, and GliZNetLoss.
"""

import pytest
from collections import namedtuple

import torch
import torch.nn as nn
from transformers import AutoModel

from gliznet.model import GliZNetConfig, GliZNetForSequenceClassification
from gliznet.model.loss import GliZNetLoss


# ──────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────────────────────────────────────


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
        **kwargs,
    ):
        batch, seq_len = input_ids.shape
        last_hidden_state = (
            input_ids.unsqueeze(-1).repeat(1, 1, self.config.hidden_size).float()
            / 1000.0
        )
        num_heads = 12
        num_layers = 12
        attentions = None
        if output_attentions:
            dummy_attention = torch.ones(batch, num_heads, seq_len, seq_len) / seq_len
            if attention_mask is not None:
                mask = (
                    attention_mask.unsqueeze(1)
                    .unsqueeze(2)
                    .expand(batch, num_heads, seq_len, seq_len)
                )
                dummy_attention = dummy_attention * mask
                dummy_attention = dummy_attention / (
                    dummy_attention.sum(dim=-1, keepdim=True) + 1e-8
                )
            attentions = tuple([dummy_attention for _ in range(num_layers)])
        if return_dict:
            return namedtuple("out", ("last_hidden_state", "attentions"))(
                last_hidden_state, attentions
            )
        return last_hidden_state


HIDDEN = 8
INPUT_IDS = torch.tensor([[101, 1012, 1013, 1014], [101, 1016, 1001, 0]])
ATTN = torch.where(INPUT_IDS > 0, 1, 0)
LMASK = torch.tensor([[0, 0, 1, 0], [0, 0, 1, 0]])
LABELS = torch.tensor([[1.0, -100], [0.0, -100]])


def _make_model(similarity_metric="cosine"):
    model = GliZNetForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        projected_dim=HIDDEN,
        similarity_metric=similarity_metric,
    )
    model.backbone = DummyEncoder(HIDDEN)
    model.config.hidden_size = HIDDEN
    model.hidden_size = HIDDEN
    model.aggregator.text_projector = nn.Identity()
    model.aggregator.label_projector = nn.Identity()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Similarity metrics
# ──────────────────────────────────────────────────────────────────────────────


class TestSimilarityMetric:
    def test_dot_config(self):
        model = _make_model("dot")
        assert model.config.similarity_metric == "dot"

    def test_bilinear_config(self):
        model = _make_model("bilinear")
        assert model.config.similarity_metric == "bilinear"
        assert hasattr(model.aggregator.similarity_head, "classifier")
        assert isinstance(model.aggregator.similarity_head.classifier, nn.Bilinear)

    def test_dot_no_classifier(self):
        model = _make_model("dot")
        assert not hasattr(model.aggregator.similarity_head, "classifier")

    @pytest.mark.parametrize("metric", ["dot", "bilinear", "cosine"])
    def test_forward_shape_no_labels(self, metric):
        out = _make_model(metric)(
            input_ids=INPUT_IDS, attention_mask=ATTN, lmask=LMASK, labels=None
        )
        assert "logits" in out
        assert out["logits"].shape == (2, 1)

    @pytest.mark.parametrize("metric", ["dot", "bilinear", "cosine"])
    def test_forward_with_labels_has_loss(self, metric):
        out = _make_model(metric)(
            input_ids=INPUT_IDS, attention_mask=ATTN, lmask=LMASK, labels=LABELS
        )
        assert "loss" in out
        assert isinstance(out["loss"], torch.Tensor)
        assert out["loss"].item() >= 0.0
        # With labels, model reconstructs dense (B, max_labels) logits
        assert out["logits"].shape == (2, 2)

    @pytest.mark.parametrize("metric", ["dot", "bilinear", "cosine"])
    def test_logits_finite(self, metric):
        out = _make_model(metric)(
            input_ids=INPUT_IDS, attention_mask=ATTN, lmask=LMASK, labels=None
        )
        assert out["logits"].isfinite().all()

    def test_invalid_metric_raises(self):
        with pytest.raises((ValueError, KeyError)):
            GliZNetForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                projected_dim=HIDDEN,
                similarity_metric="invalid_metric",
            )

    def test_deterministic_same_weights(self):
        fixed_ids = torch.tensor([[101, 1000, 2000, 3000], [101, 4000, 5000, 0]])
        fixed_attn = torch.where(fixed_ids > 0, 1, 0)
        fixed_lmask = torch.tensor([[0, 0, 1, 0], [0, 0, 1, 0]])

        m1 = _make_model("cosine")
        m2 = _make_model("cosine")
        m2.load_state_dict(m1.state_dict())

        o1 = m1(input_ids=fixed_ids, attention_mask=fixed_attn, lmask=fixed_lmask)
        o2 = m2(input_ids=fixed_ids, attention_mask=fixed_attn, lmask=fixed_lmask)
        assert torch.allclose(o1["logits"], o2["logits"], atol=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# Core forward pass
# ──────────────────────────────────────────────────────────────────────────────


class TestForwardPass:
    def test_no_labels_no_loss(self):
        model = _make_model()
        out = model(input_ids=INPUT_IDS, attention_mask=ATTN, lmask=LMASK, labels=None)
        assert "logits" in out
        assert out.loss is None

    def test_with_labels_loss_non_negative(self):
        model = _make_model()
        out = model(
            input_ids=INPUT_IDS, attention_mask=ATTN, lmask=LMASK, labels=LABELS
        )
        assert "loss" in out
        assert isinstance(out["loss"], torch.Tensor)
        assert out["loss"].item() >= 0.0

    def test_logits_shape(self):
        model = _make_model()
        out = model(input_ids=INPUT_IDS, attention_mask=ATTN, lmask=LMASK, labels=None)
        assert out["logits"].shape == (2, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Custom tokens & embedding resize
# ──────────────────────────────────────────────────────────────────────────────


class TestCustomTokens:
    @pytest.fixture
    def tokenizer(self):
        from gliznet.tokenizer import GliZNETTokenizer

        return GliZNETTokenizer.from_pretrained("bert-base-uncased", lab_token="[LAB]")

    def _make(self, tokenizer, **kwargs):
        model = GliZNetForSequenceClassification.from_pretrained(
            "bert-base-uncased", projected_dim=HIDDEN, **kwargs
        )
        model.resize_token_embeddings(len(tokenizer))
        return model

    def _swap(self, model):
        model.backbone = DummyEncoder(HIDDEN)
        model.config.hidden_size = HIDDEN
        model.aggregator.text_projector = nn.Identity()
        model.aggregator.label_projector = nn.Identity()
        return model

    def test_resize_token_embeddings(self):
        model = GliZNetForSequenceClassification.from_pretrained("bert-base-uncased")
        orig_size = model.config.backbone_config.vocab_size
        new_size = orig_size + 5
        model.resize_token_embeddings(new_size)
        assert model.config.backbone_config.vocab_size == new_size
        assert model.backbone.get_input_embeddings().num_embeddings == new_size

    def test_use_lab_token_default_false(self):
        model = GliZNetForSequenceClassification.from_pretrained("bert-base-uncased")
        assert model.config.use_lab_token_for_labels is False

    def test_use_lab_token_custom_true(self, tokenizer):
        model = GliZNetForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            use_lab_token_for_labels=True,
            lab_token_id=tokenizer.lab_token_id,
        )
        assert model.config.use_lab_token_for_labels is True
        assert model.config.lab_token_id == tokenizer.lab_token_id

    def test_forward_lab_token_mode(self, tokenizer):
        model = self._swap(
            self._make(
                tokenizer,
                use_lab_token_for_labels=True,
                lab_token_id=tokenizer.lab_token_id,
            )
        )
        batch = tokenizer.tokenize("Test text", ["positive", "negative"])
        with torch.no_grad():
            out = model(
                input_ids=batch["input_ids"].unsqueeze(0),
                attention_mask=batch["attention_mask"].unsqueeze(0),
                lmask=batch["lmask"].unsqueeze(0),
            )
        assert out.logits is not None
        assert out.logits.numel() > 0

    def test_batch_inference_scores_bounded(self, tokenizer):
        model = self._swap(self._make(tokenizer))
        texts = ["Great movie!", "Terrible film."]
        labels = [["positive", "negative"], ["good", "bad"]]
        batch = tokenizer(list(zip(texts, labels)), return_tensors="pt")
        with torch.no_grad():
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                lmask=batch["lmask"],
            )
        scores = torch.sigmoid(out["logits"])
        assert scores.numel() > 0
        assert (scores >= 0.0).all()
        assert (scores <= 1.0).all()

    @pytest.mark.parametrize("metric", ["dot", "bilinear", "cosine"])
    def test_similarity_metric_with_lab_tokens(self, tokenizer, metric):
        model = self._swap(self._make(tokenizer, similarity_metric=metric))
        batch = tokenizer.tokenize("Test text", ["positive", "negative"])
        with torch.no_grad():
            out = model(
                input_ids=batch["input_ids"].unsqueeze(0),
                attention_mask=batch["attention_mask"].unsqueeze(0),
                lmask=batch["lmask"].unsqueeze(0),
            )
        assert out.logits is not None
        assert model.config.similarity_metric == metric


# ──────────────────────────────────────────────────────────────────────────────
# Backbone weight integrity
# ──────────────────────────────────────────────────────────────────────────────


class TestBackboneWeightIntegrity:
    MODEL_NAME = "bert-base-uncased"

    @pytest.fixture(scope="class")
    def models(self):
        from gliznet.tokenizer import GliZNETTokenizer

        tokenizer = GliZNETTokenizer.from_pretrained(self.MODEL_NAME)
        config = GliZNetConfig(backbone_model=self.MODEL_NAME)
        gliznet = GliZNetForSequenceClassification.from_backbone_pretrained(
            config, tokenizer=tokenizer
        )
        gliznet.eval()
        automodel = AutoModel.from_pretrained(self.MODEL_NAME)
        automodel.eval()
        return gliznet, automodel

    def test_backbone_matches_automodel(self, models):
        gliznet, automodel = models
        ids = torch.tensor([[101, 7592, 102]])
        mask = torch.ones_like(ids)
        with torch.no_grad():
            gout = gliznet.backbone(
                input_ids=ids, attention_mask=mask, return_dict=True
            )
            aout = automodel(input_ids=ids, attention_mask=mask, return_dict=True)
        assert torch.allclose(
            gout.last_hidden_state, aout.last_hidden_state, atol=1e-5
        ), "GliZNet backbone hidden states differ from standalone AutoModel"


# ──────────────────────────────────────────────────────────────────────────────
# GliZNetLoss
# ──────────────────────────────────────────────────────────────────────────────


def _default_config(**overrides):
    cfg = GliZNetConfig(
        backbone_model="bert-base-uncased",
        bce_loss_weight=1.0,
        supcon_loss_weight=1.0,
        label_repulsion_weight=0.0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_loss_inputs(batch=2, n_labels=3, n_pos_per_sample=1):
    """Build minimal tensors to call GliZNetLoss.forward()."""
    # Each sample has n_labels label slots; assign label_ids 1..n_labels
    n_spans = batch * n_labels
    logits = torch.randn(n_spans, 1)
    batch_indices = torch.repeat_interleave(torch.arange(batch), n_labels)
    label_ids = torch.tile(torch.arange(1, n_labels + 1), (batch,))
    label_embeddings = torch.randn(n_spans, 8)
    logit_scale = torch.tensor(1.0)

    labels = torch.zeros(batch, n_labels)
    for i in range(batch):
        for j in range(n_pos_per_sample):
            labels[i, j % n_labels] = 1.0

    return logits, labels, batch_indices, label_ids, label_embeddings, logit_scale


class TestGliZNetLoss:
    def test_returns_scalar(self):
        loss_fn = GliZNetLoss(_default_config())
        args = _make_loss_inputs()
        out = loss_fn(*args)
        assert out.dim() == 0
        assert out.item() >= 0.0

    def test_non_negative(self):
        loss_fn = GliZNetLoss(_default_config())
        for _ in range(5):
            out = loss_fn(*_make_loss_inputs())
            assert out.item() >= 0.0

    def test_finite(self):
        loss_fn = GliZNetLoss(_default_config())
        out = loss_fn(*_make_loss_inputs())
        assert torch.isfinite(out)

    def test_softmax_only(self):
        loss_fn = GliZNetLoss(
            _default_config(bce_loss_weight=0.0, supcon_loss_weight=1.0)
        )
        out = loss_fn(*_make_loss_inputs())
        assert torch.isfinite(out)
        assert out.item() >= 0.0

    def test_bce_only(self):
        loss_fn = GliZNetLoss(
            _default_config(bce_loss_weight=1.0, supcon_loss_weight=0.0)
        )
        out = loss_fn(*_make_loss_inputs())
        assert torch.isfinite(out)
        assert out.item() >= 0.0

    def test_repulsion_enabled(self):
        loss_fn = GliZNetLoss(_default_config(label_repulsion_weight=0.1))
        out = loss_fn(*_make_loss_inputs(batch=2, n_labels=3))
        assert torch.isfinite(out)
        assert out.item() >= 0.0

    def test_all_losses_disabled_returns_zero(self):
        loss_fn = GliZNetLoss(
            _default_config(
                bce_loss_weight=0.0,
                supcon_loss_weight=0.0,
                label_repulsion_weight=0.0,
            )
        )
        out = loss_fn(*_make_loss_inputs())
        assert out.item() == pytest.approx(0.0)

    def test_no_positives_returns_zero_supcon(self):
        """Samples with no positive labels → softmax loss should be 0 (skipped)."""
        loss_fn = GliZNetLoss(
            _default_config(bce_loss_weight=0.0, supcon_loss_weight=1.0)
        )
        logits, labels, batch_indices, label_ids, embs, scale = _make_loss_inputs()
        labels_no_pos = torch.zeros_like(labels)
        out = loss_fn(logits, labels_no_pos, batch_indices, label_ids, embs, scale)
        assert out.item() == pytest.approx(0.0, abs=1e-6)

    def test_empty_logits_returns_zero(self):
        loss_fn = GliZNetLoss(_default_config())
        empty_logits = torch.zeros(0, 1)
        empty_labels = torch.zeros(2, 3)
        empty_batch = torch.zeros(0, dtype=torch.long)
        empty_ids = torch.zeros(0, dtype=torch.long)
        empty_embs = torch.zeros(0, 8)
        scale = torch.tensor(1.0)
        out = loss_fn(
            empty_logits, empty_labels, empty_batch, empty_ids, empty_embs, scale
        )
        assert out.item() == pytest.approx(0.0, abs=1e-6)

    def test_perfect_scores_lower_loss_than_random(self):
        """Logits that perfectly separate positives from negatives should give lower loss."""
        loss_fn = GliZNetLoss(_default_config())
        _, labels, batch_indices, label_ids, embs, scale = _make_loss_inputs(
            batch=4, n_labels=4
        )

        # Perfect scores: positives at +10, negatives at -10
        perfect_logits = torch.where(
            labels[batch_indices, label_ids - 1].unsqueeze(-1) > 0.5,
            torch.tensor(10.0),
            torch.tensor(-10.0),
        )
        random_logits = torch.randn_like(perfect_logits)

        loss_perfect = loss_fn(
            perfect_logits, labels, batch_indices, label_ids, embs, scale
        )
        loss_random = loss_fn(
            random_logits, labels, batch_indices, label_ids, embs, scale
        )

        assert loss_perfect.item() < loss_random.item()

    def test_gradients_flow(self):
        """Loss must provide gradients to logits."""
        loss_fn = GliZNetLoss(_default_config())
        logits, labels, batch_indices, label_ids, embs, scale = _make_loss_inputs()
        logits = logits.requires_grad_(True)
        out = loss_fn(logits, labels, batch_indices, label_ids, embs, scale)
        out.backward()
        assert logits.grad is not None
        assert logits.grad.isfinite().all()

    @pytest.mark.parametrize("n_labels", [1, 2, 5, 10])
    def test_various_label_counts(self, n_labels):
        loss_fn = GliZNetLoss(_default_config())
        out = loss_fn(*_make_loss_inputs(batch=2, n_labels=n_labels))
        assert torch.isfinite(out)

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_various_batch_sizes(self, batch_size):
        loss_fn = GliZNetLoss(_default_config())
        out = loss_fn(*_make_loss_inputs(batch=batch_size, n_labels=3))
        assert torch.isfinite(out)
        assert out.item() >= 0.0
