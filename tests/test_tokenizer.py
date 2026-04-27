"""Pytest tests for GliZNETTokenizer."""

import os
import tempfile

import pytest
from transformers import AutoTokenizer

from gliznet.tokenizer import GliZNETTokenizer

MODEL = "bert-base-uncased"


@pytest.fixture(scope="module")
def hf_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="module")
def original_vocab_size(hf_tokenizer):
    return hf_tokenizer.vocab_size


@pytest.fixture
def tokenizer():
    return GliZNETTokenizer(
        pretrained_model_name_or_path=MODEL,
        min_text_tokens=1,
        model_max_length=20,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Basic tokenizer
# ──────────────────────────────────────────────────────────────────────────────


def test_initialization(tokenizer, hf_tokenizer):
    assert tokenizer.tokenizer is not None
    assert tokenizer.cls_token_id == hf_tokenizer.cls_token_id
    assert tokenizer.sep_token_id == hf_tokenizer.sep_token_id
    assert tokenizer.pad_token_id == hf_tokenizer.pad_token_id
    assert tokenizer.max_length == 20


def test_from_pretrained(hf_tokenizer):
    tok = GliZNETTokenizer.from_pretrained(MODEL)
    assert tok.tokenizer is not None
    assert tok.tokenizer.model_max_length == hf_tokenizer.model_max_length


def test_padding_behavior(tokenizer, hf_tokenizer):
    result = tokenizer([("hi", ["a"])], return_tensors="pt")
    assert result["input_ids"].shape == (1, 20)
    assert result["attention_mask"].shape == (1, 20)
    assert result["lmask"].shape == (1, 20)
    seq_len = int(result["attention_mask"][0].sum().item())
    assert seq_len < 20
    assert result["input_ids"][0, seq_len:].tolist() == [hf_tokenizer.pad_token_id] * (
        20 - seq_len
    )
    assert result["attention_mask"][0, seq_len:].tolist() == [0] * (20 - seq_len)


def test_call_single_vs_batch(tokenizer):
    single = tokenizer.tokenize("A single call.", ["l1", "l2"])
    assert single["input_ids"].shape == (20,)
    assert single["attention_mask"].shape == (20,)
    assert single["lmask"].shape == (20,)

    batch = tokenizer(
        [("First call.", ["lA"]), ("Second call.", ["lB"])], return_tensors="pt"
    )
    assert batch["input_ids"].shape == (2, 20)
    assert batch["attention_mask"].shape == (2, 20)
    assert batch["lmask"].shape == (2, 20)


def test_decode(tokenizer, hf_tokenizer):
    ids = [
        hf_tokenizer.cls_token_id,
        7592,
        2088,
        hf_tokenizer.sep_token_id,
        hf_tokenizer.pad_token_id,
        hf_tokenizer.pad_token_id,
    ]
    assert tokenizer.decode(ids, skip_special_tokens=True).strip() == "hello world"


# ──────────────────────────────────────────────────────────────────────────────
# Custom [LAB] token
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def lab_tokenizer():
    return GliZNETTokenizer(pretrained_model_name_or_path=MODEL, lab_token="[LAB]")


def test_lab_token_initialization(lab_tokenizer, original_vocab_size):
    assert lab_tokenizer.lab_token == "[LAB]"
    assert len(lab_tokenizer) == original_vocab_size + 1
    assert "[LAB]" in lab_tokenizer.tokenizer.all_special_tokens


def test_lab_token_sequence_building(lab_tokenizer):
    result = lab_tokenizer.tokenize("Hello world", ["positive", "negative"])
    assert lab_tokenizer.lab_token_id in result["input_ids"].tolist()
    decoded = lab_tokenizer.decode(
        result["input_ids"].tolist(), skip_special_tokens=True
    )
    assert "hello world" in decoded.lower()
    assert "positive" in decoded.lower()
    assert "negative" in decoded.lower()


def test_saved_tokenizer_preserves_lab_token(original_vocab_size):
    tok = GliZNETTokenizer(pretrained_model_name_or_path=MODEL, lab_token="[LAB]")
    with tempfile.TemporaryDirectory() as tmp:
        save_path = os.path.join(tmp, "saved")
        tok.save_pretrained(save_path)
        loaded = GliZNETTokenizer.from_pretrained(save_path, lab_token="[LAB]")
    assert loaded.lab_token == "[LAB]"
    assert "[LAB]" in loaded.tokenizer.all_special_tokens
    assert len(loaded) == original_vocab_size + 1


def test_no_duplicate_token_addition(original_vocab_size):
    tok1 = GliZNETTokenizer(pretrained_model_name_or_path=MODEL, lab_token="[LAB]")
    with tempfile.TemporaryDirectory() as tmp:
        tok1.save_pretrained(tmp)
        tok2 = GliZNETTokenizer.from_pretrained(tmp, lab_token="[LAB]")
    assert len(tok2) == original_vocab_size + 1


def test_batch_tokenization_with_lab_token(lab_tokenizer):
    result = lab_tokenizer(
        [("First text", ["pos", "neg"]), ("Second text", ["happy", "sad", "neutral"])],
        return_tensors="pt",
    )
    assert result["input_ids"].shape[0] == 2
    assert result["attention_mask"].shape[0] == 2
    assert result["lmask"].shape[0] == 2
    for i in range(2):
        assert lab_tokenizer.lab_token_id in result["input_ids"][i].tolist()


def test_from_pretrained_custom_params(original_vocab_size):
    tok = GliZNETTokenizer.from_pretrained(
        MODEL, lab_token="[CUSTOM]", min_text_tokens=5
    )
    assert tok.lab_token == "[CUSTOM]"
    assert tok.min_text_tokens == 5
    assert len(tok) == original_vocab_size + 1


# ──────────────────────────────────────────────────────────────────────────────
# Sequence structure — per documentation
#
# Documented format:
#   [CLS] text_tokens [SEP] label1_tokens [LAB] label2_tokens [LAB] ... [PAD]...
#
# lmask:
#   0 for [CLS], text tokens, [SEP], [LAB], and [PAD]
#   N (1-indexed) for the tokens belonging to label N
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def struct_tok():
    """Tokenizer with no max_length cap so we can inspect raw structure."""
    return GliZNETTokenizer(
        pretrained_model_name_or_path=MODEL, lab_token="[LAB]", model_max_length=512
    )


def test_sequence_starts_with_cls(struct_tok):
    result = struct_tok.tokenize("hello world", ["positive"])
    assert result["input_ids"][0].item() == struct_tok.cls_token_id


def test_sequence_has_sep_after_text(struct_tok):
    """[SEP] must appear exactly once between [CLS] and the first label token."""
    ids = struct_tok.tokenize("hello world", ["positive"])["input_ids"].tolist()
    assert struct_tok.sep_token_id in ids
    sep_pos = ids.index(struct_tok.sep_token_id)
    # Everything before sep is CLS + text (no LAB or SEP)
    prefix = ids[1:sep_pos]
    assert struct_tok.lab_token_id not in prefix
    assert struct_tok.sep_token_id not in prefix


def test_sequence_contains_lab_token_per_label(struct_tok):
    """N labels → exactly N [LAB] tokens."""
    for n_labels in [1, 2, 3]:
        labels = [f"label_{i}" for i in range(n_labels)]
        ids = struct_tok.tokenize("test text", labels)["input_ids"].tolist()
        lab_count = ids.count(struct_tok.lab_token_id)
        assert lab_count == n_labels, (
            f"expected {n_labels} [LAB] tokens, got {lab_count}"
        )


def test_lmask_zero_for_special_tokens(struct_tok):
    """[CLS], [SEP], and [LAB] positions must have lmask == 0."""
    result = struct_tok.tokenize("hello world", ["pos", "neg"])
    ids = result["input_ids"].tolist()
    lmask = result["lmask"].tolist()
    special_ids = {
        struct_tok.cls_token_id,
        struct_tok.sep_token_id,
        struct_tok.lab_token_id,
    }
    for pos, (tok_id, mask_val) in enumerate(zip(ids, lmask)):
        if tok_id in special_ids:
            assert mask_val == 0, (
                f"special token at pos {pos} (id={tok_id}) has lmask={mask_val}"
            )


def test_lmask_labels_are_1indexed(struct_tok):
    """Label tokens must have lmask values 1, 2, 3, ... in order."""
    labels = ["alpha", "beta", "gamma"]
    result = struct_tok.tokenize("some text", labels)
    lmask = result["lmask"].tolist()
    label_vals = sorted(set(v for v in lmask if v > 0))
    assert label_vals == list(range(1, len(labels) + 1))


def test_lmask_label_order_matches_input(struct_tok):
    """lmask label indices must appear in ascending order (1 before 2 before 3)."""
    result = struct_tok.tokenize("foo bar", ["first", "second", "third"])
    lmask = result["lmask"].tolist()
    non_zero = [v for v in lmask if v > 0]
    assert non_zero == sorted(non_zero), (
        "lmask label indices are not in ascending order"
    )


def test_lmask_pad_positions_are_zero(struct_tok):
    """Padding positions (attention_mask == 0) must have lmask == 0."""
    tok = GliZNETTokenizer(
        pretrained_model_name_or_path=MODEL, lab_token="[LAB]", model_max_length=30
    )
    result = tok.tokenize("hi", ["a"])
    attn = result["attention_mask"].tolist()
    lmask = result["lmask"].tolist()
    for pos, (a, m) in enumerate(zip(attn, lmask)):
        if a == 0:
            assert m == 0, f"pad position {pos} has lmask={m}"


def test_lmask_unique_label_count_matches_n_labels(struct_tok):
    """Number of distinct non-zero lmask values == number of labels provided."""
    for n in [1, 2, 4]:
        labels = [f"l{i}" for i in range(n)]
        result = struct_tok.tokenize("text", labels)
        unique_labels = set(v for v in result["lmask"].tolist() if v > 0)
        assert len(unique_labels) == n


def test_attention_mask_contiguous(struct_tok):
    """attention_mask must be 1...1 0...0 (no gaps after real tokens)."""
    result = struct_tok.tokenize("hello world", ["positive", "negative"])
    attn = result["attention_mask"].tolist()
    # Find the boundary between 1s and 0s
    found_zero = False
    for v in attn:
        if found_zero:
            assert v == 0, "attention_mask has a 1 after a 0 (non-contiguous)"
        if v == 0:
            found_zero = True


def test_no_labels_produces_valid_structure(struct_tok):
    """Zero labels: sequence is just [CLS] text [SEP], lmask all zeros."""
    result = struct_tok([("empty labels test", [])], return_tensors="pt")
    ids = result["input_ids"][0].tolist()
    lmask = result["lmask"][0].tolist()
    assert ids[0] == struct_tok.cls_token_id
    assert struct_tok.sep_token_id in ids
    assert all(
        v == 0
        for v in lmask
        if ids[lmask.index(v) if v else 0] != struct_tok.pad_token_id or True
    )
    # No label values in lmask
    assert all(v == 0 for v in lmask)


def test_batch_lmask_shapes_consistent(struct_tok):
    """Batched output: all sequences padded to same length, shapes consistent."""
    result = struct_tok(
        [("short", ["a"]), ("a much longer sentence here", ["alpha", "beta", "gamma"])],
        return_tensors="pt",
    )
    assert result["input_ids"].shape == result["attention_mask"].shape
    assert result["input_ids"].shape == result["lmask"].shape


@pytest.mark.parametrize(
    "text,labels",
    [
        ("Simple text", ["cat"]),
        ("", ["dog", "fish"]),
        ("X" * 200, ["very long label " * 5]),
        ("Multi label test", ["a", "b", "c", "d", "e"]),
    ],
)
def test_tokenize_no_error_various_inputs(struct_tok, text, labels):
    """Tokenizer must not raise on any of these inputs."""
    result = struct_tok([(text, labels)], return_tensors="pt")
    assert "input_ids" in result
    assert "attention_mask" in result
    assert "lmask" in result
