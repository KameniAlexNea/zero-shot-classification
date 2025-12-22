import functools
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoTokenizer, BertTokenizer


class GliZNETTokenizer:
    """Simple zero-shot classification tokenizer.

    Builds sequences like: [CLS] text [SEP] label1_tokens [LAB] label2_tokens [LAB] ...
    Each label's tokens get assigned a unique ID in lmask (1, 2, 3, etc.)
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = "bert-base-uncased",
        lab_token: str = "[LAB]",
        max_length: int = 512,
        **kwargs,
    ):
        """Initialize GliZNET tokenizer.

        Args:
            pretrained_model_name_or_path: HuggingFace model identifier or path
            lab_token: Label separator token (default: '[LAB]')
            max_length: Maximum sequence length
        """
        self.tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )

        self.lab_token = lab_token

        # Add label token if it doesn't exist
        additional_tokens = getattr(self.tokenizer, "additional_special_tokens", [])
        if lab_token not in additional_tokens:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [lab_token]}
            )

        # Set max_length
        model_max = self.tokenizer.model_max_length
        self.max_length = (
            max_length if model_max > 100000 else min(model_max, max_length)
        )

        # Cache token IDs
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.lab_token_id = self.tokenizer.convert_tokens_to_ids(lab_token)

        # Create cached label tokenization
        @functools.lru_cache(maxsize=10000)
        def _tokenize_label_cached(label: str) -> Tuple[int, ...]:
            return tuple(
                self.tokenizer.encode(label, add_special_tokens=False, truncation=False)
            )

        self._tokenize_label_cached = _tokenize_label_cached

    def _build_sequence(
        self, text: str, labels: List[str]
    ) -> Tuple[List[int], List[int]]:
        """Build sequence: [CLS] text [SEP] label1 [LAB] label2 [LAB] ..."""
        # Tokenize text (no special tokens, no truncation warnings)
        text_ids = self.tokenizer.encode(
            text, add_special_tokens=False, truncation=True, max_length=self.max_length
        )

        # Tokenize all labels (cached, truncate individual labels)
        label_ids_list = [
            list(
                self.tokenizer.encode(
                    label, add_special_tokens=False, truncation=True, max_length=128
                )
            )
            for label in labels
        ]

        # Calculate space: [CLS] + text + [SEP] + labels + [LAB] separators
        overhead = 2  # [CLS] and [SEP]
        labels_size = sum(len(ids) for ids in label_ids_list) + len(label_ids_list)

        # Allocate space between text and labels
        total_content = len(text_ids) + labels_size
        if total_content + overhead > self.max_length:
            # Need to truncate
            available = self.max_length - overhead
            text_budget = max(available // 2, 10)  # Give at least 10 tokens to text
            label_budget = available - text_budget

            text_ids = text_ids[:text_budget]

            # Fit as many complete labels as possible
            fitted_labels = []
            used = 0
            for label_ids in label_ids_list:
                needed = len(label_ids) + 1  # +1 for [LAB] separator
                if used + needed <= label_budget:
                    fitted_labels.append(label_ids)
                    used += needed
                else:
                    break
            label_ids_list = fitted_labels

        # Build sequence
        sequence = [self.cls_token_id] + text_ids + [self.sep_token_id]
        lmask = [0] * len(sequence)

        # Add each label with its ID
        for label_idx, label_ids in enumerate(label_ids_list, start=1):
            # All tokens from this label get the same ID
            for token_id in label_ids:
                sequence.append(token_id)
                lmask.append(label_idx)
            # Add separator (not part of label representation)
            sequence.append(self.lab_token_id)
            lmask.append(0)

        return sequence, lmask

    def __call__(
        self,
        examples: List[Tuple[str, List[str]]],
        return_tensors: str = "pt",
    ) -> Dict[str, Any]:
        """Tokenize batch of (text, labels) tuples.

        Args:
            examples: List of (text, list_of_labels) tuples
            return_tensors: "pt" for PyTorch tensors, None for lists

        Returns:
            Dictionary with input_ids, attention_mask, and lmask
        """
        # Build all sequences
        all_sequences = [
            self._build_sequence(text, labels) for text, labels in examples
        ]
        sequences, lmasks = zip(*all_sequences)

        # Find max length in batch
        max_len = min(max(len(seq) for seq in sequences), self.max_length)

        # Pad all sequences
        input_ids = []
        attention_mask = []
        padded_lmasks = []

        for seq, lmask in zip(sequences, lmasks):
            # Truncate if needed
            seq = seq[:max_len]
            lmask = lmask[:max_len]

            # Pad
            pad_len = max_len - len(seq)
            input_ids.append(seq + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(seq) + [0] * pad_len)
            padded_lmasks.append(lmask + [0] * pad_len)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "lmask": padded_lmasks,
        }

        if return_tensors == "pt":
            result = {k: torch.tensor(v, dtype=torch.long) for k, v in result.items()}

        return result

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load tokenizer from pretrained model."""
        return cls(pretrained_model_name_or_path, **kwargs)

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save the underlying tokenizer."""
        return self.tokenizer.save_pretrained(save_directory, **kwargs)

    def push_to_hub(
        self,
        repo_id: str,
        private: bool = False,
        commit_message: str = "Upload GliZNET tokenizer",
        **kwargs,
    ):
        """Push the underlying tokenizer to Hugging Face Hub."""
        return self.tokenizer.push_to_hub(
            repo_id, private=private, commit_message=commit_message, **kwargs
        )

    def decode(self, token_ids: List[int], **kwargs) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, **kwargs)

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)

    def __len__(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)
