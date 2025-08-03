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
    truncation=True,
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

        # Add custom [LAB] token if not using default ";" separator and token doesn't exist
        if (
            cls_separator_token != ";"
            and cls_separator_token not in additional_special_tokens
        ):
            # Add the custom token to the tokenizer
            special_tokens_dict = {"additional_special_tokens": [cls_separator_token]}
            self.tokenizer.add_special_tokens(special_tokens_dict)

        self.max_length = self.tokenizer.model_max_length
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mask_token_id = self.tokenizer.mask_token_id or self.tokenizer.unk_token_id
        self.label_sep_id = self.tokenizer.convert_tokens_to_ids(
            cls_separator_token
        )  # ';' is used as label separator. Try [SEP] if you want to use it as label separator.

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

    def _build_full_sequence_text(self, text: str, labels: List[str]) -> str:
        """Build the full sequence as text: [CLS] + text + [SEP] + label1 + [LAB] + label2 + [LAB] + ..."""
        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token

        # More efficient string building
        if not labels:
            return f"{cls_token} {text} {sep_token}"

        # Each label ends with its separator token (acts as the label embedding)
        labels_with_sep = [f"{label} {self.cls_separator_token}" for label in labels]
        labels_part = " ".join(labels_with_sep)
        return f"{cls_token} {text} {sep_token} {labels_part}"

    def _build_reverse_sequence_text(self, text: str, labels: List[str]) -> str:
        """Build the reverse sequence for truncation: labels + [SEP] + text"""
        sep_token = self.tokenizer.sep_token

        # More efficient string building
        if not labels:
            return f"{sep_token} {text}"

        # Each label ends with its separator token (acts as the label embedding)
        labels_with_sep = [f"{label} {self.cls_separator_token}" for label in labels]
        labels_part = " ".join(labels_with_sep)
        return f"{labels_part} {sep_token} {text}"

    def _tokenize_with_truncation_strategy(
        self, text: str, labels: List[str]
    ) -> tuple[List[int], List[int]]:
        """Tokenize using reverse strategy to prioritize text over labels when truncating."""
        # First, try to build the full sequence
        full_sequence = self._build_full_sequence_text(text, labels)
        full_tokens = self.tokenizer(full_sequence, **tokenizer_config)["input_ids"]

        # If it fits, return as is
        if len(full_tokens) <= self.max_length:
            return self._parse_tokenized_sequence(full_tokens, labels)

        # If it doesn't fit, use reverse tokenization strategy
        reverse_sequence = self._build_reverse_sequence_text(text, labels)

        # Tokenize the reverse sequence with truncation
        truncated_config = {**tokenizer_config, "max_length": self.max_length}
        reverse_tokens = self.tokenizer(reverse_sequence, **truncated_config)[
            "input_ids"
        ]

        # Now we need to rearrange the tokens to get the correct order
        # Find [SEP] token in reverse sequence
        try:
            sep_idx = reverse_tokens.index(self.sep_token_id)
        except ValueError:
            # If no SEP found, fallback to original strategy
            return self._parse_tokenized_sequence(
                full_tokens[: self.max_length], labels
            )

        # Extract text tokens (after [SEP] in reverse) and labels (before [SEP] in reverse)
        text_tokens = reverse_tokens[sep_idx + 1 :]  # Text comes after [SEP] in reverse
        label_tokens = reverse_tokens[:sep_idx]  # Labels come before [SEP] in reverse

        # Reconstruct the proper sequence: [CLS] + text + [SEP] + labels
        proper_sequence = (
            [self.cls_token_id] + text_tokens + [self.sep_token_id] + label_tokens
        )

        return self._parse_tokenized_sequence(proper_sequence, labels)

    def _parse_tokenized_sequence(
        self, token_ids: List[int], labels: List[str]
    ) -> tuple[List[int], List[int]]:
        """Parse the tokenized sequence to extract text tokens and create label mask."""
        # Create label mask with same length as token_ids
        lmask = [0] * len(token_ids)

        # Find [SEP] token position - only search once
        try:
            sep_idx = token_ids.index(self.sep_token_id)
        except ValueError:
            # If no SEP found, assume everything after [CLS] is text
            return token_ids, lmask

        if self.cls_separator_token != ";":
            label_idx = 1
            for i, token_id in enumerate(token_ids[sep_idx + 1 :], start=sep_idx + 1):
                if token_id == self.label_sep_id:
                    lmask[i] = label_idx
                    label_idx += 1
        else:

            # Mark label tokens in the mask - iterate directly over tokens after [SEP]
            label_idx = 1

            # Iterate through tokens after [SEP]
            for i, token_id in enumerate(token_ids[sep_idx + 1 :], start=sep_idx + 1):
                if token_id == self.label_sep_id:
                    label_idx += 1
                else:
                    # This is a label token
                    lmask[i] = label_idx

        return token_ids, lmask

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
        # Pre-build all sequences in one go
        sequences = [
            self._build_full_sequence_text(text, labels)
            for text, labels in zip(texts, all_labels)
        ]

        # Tokenize all sequences at once
        tokenizer_config = {
            "padding": False,
            "truncation": False,
            "return_tensors": None,
            "add_special_tokens": False,
        }
        batch_tokens = self.tokenizer(sequences, **tokenizer_config)["input_ids"]

        # Process each sequence
        input_ids_batch = []
        attention_mask_batch = []
        lmask_batch = []

        for i, (tokens, labels) in enumerate(zip(batch_tokens, all_labels)):
            # Check if truncation is needed
            if len(tokens) > self.max_length:
                # Use reverse strategy for truncation
                token_ids, lmask = self._tokenize_with_truncation_strategy(
                    texts[i], labels
                )
            else:
                # Parse normally
                token_ids, lmask = self._parse_tokenized_sequence(tokens, labels)

            # Apply token dropout if specified
            if token_dropout > 0.0:
                token_ids = apply_token_dropout(
                    token_ids, token_dropout, self.mask_token_id
                )

            # Pad
            pad_len = self.max_length - len(token_ids)
            input_ids = token_ids + [self.pad_token_id] * pad_len
            attention_mask = [1] * len(token_ids) + [0] * pad_len
            lmask = lmask + [0] * pad_len

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            lmask_batch.append(lmask)

        result = {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "lmask": lmask_batch,
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

    def _to_tensors(self, results: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return {
            k: torch.tensor(
                results[k],
                dtype=torch.long,
            )
            for k in ("input_ids", "attention_mask", "lmask")
        }
