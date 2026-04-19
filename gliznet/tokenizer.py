import functools
from typing import Dict, List, Tuple, Union

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
        max_tokens_per_span: int = 64,
        min_text_tokens: int = 10,
        min_label_tokens: int = 2,
        **kwargs,
    ):
        """Initialize GliZNET tokenizer.

        Args:
            pretrained_model_name_or_path: HuggingFace model identifier or path
            lab_token: Label separator token (default: '[LAB]')
            max_tokens_per_span: Maximum number of tokens per label span
            min_text_tokens: Minimum number of tokens reserved for text when truncating
            min_label_tokens: Minimum number of tokens kept per label when truncating
            **kwargs: Forwarded to AutoTokenizer.from_pretrained, e.g. model_max_length=512
        """
        self.tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )

        self.lab_token = lab_token
        self.max_tokens_per_span = max_tokens_per_span
        self.min_text_tokens = min_text_tokens
        self.min_label_tokens = min_label_tokens

        # Add label token if it doesn't exist
        additional_tokens = getattr(self.tokenizer, "additional_special_tokens", [])
        if lab_token not in additional_tokens:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [lab_token]}
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
                self.tokenizer.encode(
                    label,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_tokens_per_span,
                )
            )

        self._tokenize_label_cached = _tokenize_label_cached
        self.tokenizer.init_kwargs["max_tokens_per_span"] = self.max_tokens_per_span
        self.tokenizer.init_kwargs["min_text_tokens"] = self.min_text_tokens
        self.tokenizer.init_kwargs["min_label_tokens"] = self.min_label_tokens

    @property
    def max_length(self) -> int:
        return self.tokenizer.model_max_length or 1_000_000

    def _build_sequence(
        self, text: str, labels: List[str]
    ) -> Tuple[List[int], List[int]]:
        """Build sequence: [CLS] text [SEP] label1 [LAB] label2 [LAB] ..."""
        # Tokenize text (no special tokens, no truncation warnings)
        text_ids = self.tokenizer.encode(
            text, add_special_tokens=False, truncation=True, max_length=self.max_length
        )

        # Tokenize all labels via cache, truncated to max_tokens_per_span
        label_ids_list = [list(self._tokenize_label_cached(label)) for label in labels]

        # Calculate space: [CLS] + text + [SEP] + labels + [LAB] separators
        overhead = 2  # [CLS] and [SEP]
        n_labels = len(label_ids_list)
        labels_size = (
            sum(len(ids) for ids in label_ids_list) + n_labels
        )  # +1 [LAB] per label

        # Allocate space between text and labels
        total_content = len(text_ids) + labels_size
        if total_content + overhead > self.max_length:
            available = self.max_length - overhead

            if n_labels == 0:
                # No labels: just truncate text
                text_ids = text_ids[:available]
            else:
                # Step 1: reduce labels, text untouched.
                # Each label is guaranteed min_label_tokens; the remaining surplus
                # is distributed proportionally to each label's original length so
                # that short labels lose little and long labels absorb the bulk of
                # the cut.
                label_content_budget = available - len(text_ids) - n_labels

                if label_content_budget >= n_labels * self.min_label_tokens:
                    total_original = sum(len(ids) for ids in label_ids_list)
                    surplus = label_content_budget - n_labels * self.min_label_tokens
                    label_ids_list = [
                        ids[
                            : self.min_label_tokens
                            + (
                                int(surplus * len(ids) / total_original)
                                if total_original > 0
                                else 0
                            )
                        ]
                        for ids in label_ids_list
                    ]
                else:
                    # Step 2: even min_label_tokens per label overflows — truncate text too
                    label_ids_list = [
                        ids[: self.min_label_tokens] for ids in label_ids_list
                    ]
                    min_labels_size = n_labels * self.min_label_tokens + n_labels
                    text_budget = max(available - min_labels_size, self.min_text_tokens)
                    text_ids = text_ids[:text_budget]

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

    def tokenize(
        self,
        texts: Union[str, List[str]],
        text_labels: Union[List[str], List[List[str]]],
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        if isinstance(texts, str) and not isinstance(text_labels[0], str):
            raise ValueError(
                "If 'texts' is a string, 'text_labels' must be a list of strings."
            )
        is_unique = False
        if isinstance(texts, str):
            texts = [texts]
            text_labels = [text_labels]
            is_unique = True
        examples = list(zip(texts, text_labels))
        results = self.__call__(examples, return_tensors=return_tensors)
        if is_unique:
            results = {k: v[0] for k, v in results.items()}
        return results

    def __call__(
        self,
        examples: List[Tuple[str, List[str]]],
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
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

        # Pad to model_max_length when set, otherwise to the longest sequence in the batch
        model_max = self.tokenizer.model_max_length
        max_len = (
            model_max
            if (model_max and model_max <= 1_000_000)
            else max(len(seq) for seq in sequences)
        )

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
