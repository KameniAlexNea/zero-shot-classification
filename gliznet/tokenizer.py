import functools
import random
from typing import Any, Dict, List, Union

import torch
from transformers import AutoTokenizer, BertTokenizerFast


def apply_token_dropout(
    tokens: List[int], dropout_prob: float, mask_token_id: int
) -> List[int]:
    return [tok if random.random() > dropout_prob else mask_token_id for tok in tokens]


tokenizer_config = dict(
    return_attention_mask=False,
    return_token_type_ids=False,
    add_special_tokens=False,
)


class GliZNETTokenizer:
    def __init__(
        self,
        pretrained_model_name_or_path: str = "bert-base-uncased",
        min_text_token: int = 5,
        cls_separator_token: str = "[LAB]",  # ;
        *args,
        **kwargs,
    ):
        """Initialize GliZNET tokenizer.

        Args:
            pretrained_model_name_or_path: HuggingFace model identifier or path
            min_text_token: Minimum number of text tokens (unused, kept for compatibility)
            cls_separator_token: Token to separate labels. Use '[LAB]' for separator pooling
                or ';' for mean pooling over label content tokens

        Pooling Strategies:
            - Separator pooling (cls_separator_token='[LAB]'): Uses the separator token
              embedding directly. Marks separator positions in lmask.
            - Mean pooling (cls_separator_token=';'): Averages embeddings of label content
              tokens. Marks content token positions in lmask.
        """
        self.tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )

        self.min_text_token = min_text_token

        # Check if tokenizer already has custom tokens and detect [LAB] token
        additional_special_tokens = getattr(
            self.tokenizer, "additional_special_tokens", []
        )
        self._auto_detected = False

        if additional_special_tokens:
            # Look for [LAB] token or similar custom separator tokens
            lab_tokens = [
                token
                for token in additional_special_tokens
                if token.startswith("[") and token.endswith("]")
            ]
            if lab_tokens:
                # Use the first bracket token found as the separator
                detected_separator = lab_tokens[0]
                if cls_separator_token == ";" and detected_separator != ";":
                    # Auto-detect and use the found custom token
                    cls_separator_token = detected_separator
                    self._auto_detected = True

        self.cls_separator_token = cls_separator_token

        # Determine pooling strategy based on separator token
        self.pooling_strategy = "separator" if cls_separator_token != ";" else "mean"

        # Add custom [LAB] token if not using default ";" separator and token doesn't exist
        if (
            cls_separator_token != ";"
            and cls_separator_token not in additional_special_tokens
        ):
            # Add the custom token to the tokenizer
            special_tokens_dict = {"additional_special_tokens": [cls_separator_token]}
            self.tokenizer.add_special_tokens(special_tokens_dict)

        # Ensure max_length is reasonable (deberta can have very large default)
        model_max_length = self.tokenizer.model_max_length
        if model_max_length > 100000:  # Likely unset, use 512 as default
            model_max_length = 512
        self.max_length = model_max_length
        
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mask_token_id = self.tokenizer.mask_token_id or self.tokenizer.unk_token_id
        self.label_sep_id = self.tokenizer.convert_tokens_to_ids(cls_separator_token)

        # Create LRU-cached tokenization function
        @functools.lru_cache(maxsize=10000)
        def _tokenize_label_cached(label: str) -> tuple:
            tokens = self.tokenizer(label, **tokenizer_config)["input_ids"]
            return tuple(tokens + [self.label_sep_id])

        self._tokenize_label_cached = _tokenize_label_cached

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str = "bert-base-uncased",
        min_text_token=5,
        cls_separator_token: str = "[LAB]",
        *args,
        **kwargs,
    ) -> "GliZNETTokenizer":
        return cls(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            min_text_token=min_text_token,
            cls_separator_token=cls_separator_token,
            *args,
            **kwargs,
        )

    def _pad_and_mask(self, token_ids: List[int], lmask: List[int]) -> Dict[str, Any]:
        # Truncate both sequences to max_length if needed
        if len(token_ids) > self.max_length:
            token_ids = token_ids[: self.max_length]
            lmask = lmask[: self.max_length]

        pad_len = self.max_length - len(token_ids)
        input_ids = token_ids + [self.pad_token_id] * pad_len
        attention_mask = [1] * len(token_ids) + [0] * pad_len
        lmask = lmask + [0] * pad_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "lmask": lmask,
        }

    def _prepare_label_sequences(self, labels: List[str]) -> List[List[int]]:
        """Retrieve cached label token sequences using LRU cache."""
        return [list(self._tokenize_label_cached(label)) for label in labels]

    def _assemble_sequence(
        self, text_tokens: List[int], label_token_seqs: List[List[int]]
    ) -> tuple[List[int], List[int], int]:
        """Truncate text/label tokens to max length and build lmask."""
        max_label_capacity = max(self.max_length - 2, 0)
        trimmed_sequences: List[List[int]] = []
        running_len = 0
        for seq in label_token_seqs:
            seq_len = len(seq)
            if running_len + seq_len > max_label_capacity:
                break
            trimmed_sequences.append(seq)
            running_len += seq_len

        text_budget = max(self.max_length - 2 - running_len, 0)
        truncated_text = text_tokens[:text_budget]

        tokens = [self.cls_token_id, *truncated_text, self.sep_token_id]
        lmask = [0] * len(tokens)
        current_idx = len(tokens)

        if self.pooling_strategy == "separator":
            for label_idx, seq in enumerate(trimmed_sequences, 1):
                seq_len = len(seq)
                if seq_len == 0:
                    continue
                tokens.extend(seq)
                lmask.extend([0] * seq_len)
                lmask[current_idx + seq_len - 1] = label_idx
                current_idx += seq_len
        else:
            for label_idx, seq in enumerate(trimmed_sequences, 1):
                seq_len = len(seq)
                if seq_len == 0:
                    continue
                tokens.extend(seq)
                lmask.extend([0] * seq_len)
                content_len = max(seq_len - 1, 0)
                for offset in range(content_len):
                    lmask[current_idx + offset] = label_idx
                current_idx += seq_len

        return tokens, lmask, len(trimmed_sequences)

    def _tokenize_with_truncation_strategy(
        self, text: str, labels: List[str]
    ) -> tuple[List[int], List[int]]:
        """Tokenize text/labels and truncate text tokens before appending labels."""
        text_tokens = self.tokenizer(text, **tokenizer_config)["input_ids"]
        label_token_seqs = self._prepare_label_sequences(labels)
        tokens, lmask, _ = self._assemble_sequence(text_tokens, label_token_seqs)
        return tokens, lmask

    def tokenize_example(
        self,
        text: str,
        all_labels: List[str],
        token_dropout: float = 0.0,
    ) -> Dict[str, Any]:
        # Use new refactored method
        token_ids, lmask = self._tokenize_with_truncation_strategy(text, all_labels)

        # Apply token dropout if specified
        if token_dropout > 0.0:
            token_ids = apply_token_dropout(
                token_ids, token_dropout, self.mask_token_id
            )

        result = self._pad_and_mask(token_ids, lmask)
        result.update(self._to_tensors(result))
        return result

    def tokenize_batch(
        self,
        texts: List[str],
        all_labels: List[List[str]],
        token_dropout: float = 0.0,
    ) -> Dict[str, Any]:
        batch_text_tokens = self.tokenizer(texts, **tokenizer_config)["input_ids"]

        input_ids_batch = []
        attention_mask_batch = []
        lmask_batch = []
        num_labels_fitted = []

        for text_tokens, labels in zip(batch_text_tokens, all_labels):
            label_token_seqs = self._prepare_label_sequences(labels)
            token_ids, lmask, fitted = self._assemble_sequence(
                text_tokens, label_token_seqs
            )

            if token_dropout > 0.0:
                token_ids = apply_token_dropout(
                    token_ids, token_dropout, self.mask_token_id
                )

            num_labels_fitted.append(fitted)

            pad_len = self.max_length - len(token_ids)
            input_ids = token_ids + [self.pad_token_id] * pad_len
            attention_mask = [1] * len(token_ids) + [0] * pad_len
            padded_lmask = lmask + [0] * pad_len

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            lmask_batch.append(padded_lmask)

        result = {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "lmask": lmask_batch,
            "num_labels_fitted": num_labels_fitted,
        }
        result.update(self._to_tensors(result))
        return result

    def __call__(
        self,
        texts: Union[List[str], str],
        labels: List[Union[List[str], str]],
        token_dropout: float = 0.0,
    ):
        if isinstance(texts, str):
            return self.tokenize_example(texts, labels, token_dropout)
        return self.tokenize_batch(texts, labels, token_dropout)

    def decode_sequence(self, input_ids: List[int]) -> str:
        return self.tokenizer.decode(
            [i for i in input_ids if i != self.pad_token_id], skip_special_tokens=True
        )

    def get_vocab_size(self) -> int:
        return len(self.tokenizer)

    def has_custom_tokens(self) -> bool:
        """Check if custom tokens were added to the tokenizer."""
        return self.cls_separator_token != ";"

    def get_added_tokens_count(self) -> int:
        """Get the number of tokens added to the original vocabulary."""
        additional_special_tokens = getattr(
            self.tokenizer, "additional_special_tokens", []
        )
        return len(additional_special_tokens) if additional_special_tokens else 0

    def get_additional_special_tokens(self) -> List[str]:
        """Get the list of additional special tokens in the tokenizer."""
        return getattr(self.tokenizer, "additional_special_tokens", [])

    def was_auto_detected(self) -> bool:
        """Check if the cls_separator_token was auto-detected from existing tokens."""
        return hasattr(self, "_auto_detected") and self._auto_detected

    def get_pooling_strategy(self) -> str:
        """Get the pooling strategy used by this tokenizer.

        Returns:
            'separator': Uses separator token embeddings directly
            'mean': Averages label content token embeddings
        """
        return self.pooling_strategy

    def _to_tensors(self, results: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return {
            k: torch.tensor(
                results[k],
                dtype=torch.long,
            )
            for k in ("input_ids", "attention_mask", "lmask")
        }
