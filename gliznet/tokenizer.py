import random
from typing import Any, Dict, List, Optional, Union

import datasets
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast


class GliZNETTokenizer:
    def __init__(
        self,
        pretrained_model_name_or_path: str = "bert-base-uncased",
        min_text_token: int = 5,
        *args,
        **kwargs,
    ):
        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )
        self.min_text_token = min_text_token

        # Cache tokenizer properties
        self.max_length = self.tokenizer.model_max_length
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.pad_token = self.tokenizer.pad_token
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str = "bert-base-uncased",
        min_text_token=5,
        *args,
        **kwargs,
    ) -> "GliZNETTokenizer":
        return cls(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            min_text_token=min_text_token,
            *args,
            **kwargs,
        )

    def _prepare_text_with_labels(self, text: str, labels: List[str]) -> str:
        """Prepare a single text with labels separated by SEP tokens."""
        if not labels:
            return text

        # Join text and labels with SEP tokens
        labels_str = f" {self.sep_token} ".join(labels)
        combined_text = f"{text} {self.sep_token} {labels_str}"
        return combined_text

    def _calculate_label_positions(
        self, text: str, labels: List[str], input_ids: List[int]
    ) -> List[bool]:
        """Calculate positions where label tokens start after tokenization."""
        if not labels:
            return [False] * len(input_ids)

        # Tokenize text only to find where labels start
        text_only = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]

        # Account for CLS token at the beginning
        text_length_with_cls = len(text_only) + 1

        # Initialize mask
        label_mask = [False] * len(input_ids)

        # Find label positions
        current_pos = text_length_with_cls + 1  # +1 for SEP after text

        for label in labels:
            if current_pos < len(input_ids):
                label_mask[current_pos] = True

                # Tokenize label to find its length
                label_tokens = self.tokenizer(
                    label,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )["input_ids"]

                # Move to next label position (+1 for SEP between labels)
                current_pos += len(label_tokens) + 1

        return label_mask

    def _truncate_if_needed(self, text: str, labels: List[str]) -> str:
        """Truncate text if the combined sequence is too long."""
        combined = self._prepare_text_with_labels(text, labels)

        # Quick tokenization to check length
        temp_encoding = self.tokenizer(
            combined,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
            truncation=False,
            padding=False,
        )

        if len(temp_encoding["input_ids"]) <= self.max_length:
            return text

        # Calculate how much to truncate
        labels_text = f" {self.sep_token} ".join(labels) if labels else ""
        label_tokens_approx = len(
            self.tokenizer(
                labels_text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
        )

        # Reserve space for: CLS + text + SEP + labels + potential SEPs
        reserved_space = 2 + label_tokens_approx + len(labels)  # rough estimate
        max_text_length = max(self.min_text_token, self.max_length - reserved_space)

        # Truncate text
        text_tokens = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]

        if len(text_tokens) > max_text_length:
            truncated_tokens = text_tokens[:max_text_length]
            text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)

        return text

    def tokenize_example(
        self,
        text: str,
        all_labels: List[str],
        return_tensors: Optional[str] = "pt",
        pad: bool = True,
    ) -> Dict[str, Any]:
        """Tokenize a single example using transformers tokenizer."""
        # Truncate text if needed
        truncated_text = self._truncate_if_needed(text, all_labels)

        # Prepare combined text
        combined_text = self._prepare_text_with_labels(truncated_text, all_labels)

        # Use transformers tokenizer
        encoding = self.tokenizer(
            combined_text,
            add_special_tokens=True,
            max_length=self.max_length if pad else None,
            padding="max_length" if pad else False,
            truncation=True,
            return_tensors=None,  # We'll handle tensor conversion ourselves
            return_attention_mask=True,
            return_token_type_ids=False,
        )

        # Calculate label mask
        label_mask = self._calculate_label_positions(
            truncated_text, all_labels, encoding["input_ids"]
        )

        # Ensure label_mask has the same length as input_ids
        if len(label_mask) != len(encoding["input_ids"]):
            # Pad or truncate label_mask to match
            if len(label_mask) < len(encoding["input_ids"]):
                label_mask.extend([False] * (len(encoding["input_ids"]) - len(label_mask)))
            else:
                label_mask = label_mask[: len(encoding["input_ids"])]

        result = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "label_mask": label_mask,
        }

        if return_tensors == "pt":
            result = {
                k: torch.tensor(
                    v, dtype=torch.long if k != "label_mask" else torch.bool
                )
                for k, v in result.items()
            }

        return result

    def tokenize_batch(
        self,
        texts: List[str],
        all_labels: List[List[str]],
        return_tensors: Optional[str] = "pt",
        pad: bool = True,
    ) -> Dict[str, Any]:
        """Tokenize a batch of examples."""
        if return_tensors == "pt" and not pad:
            raise ValueError(
                "return_tensors='pt' requires pad=True for batch tokenization."
            )

        # Process each example
        batch_results = []
        for text, labels in zip(texts, all_labels):
            result = self.tokenize_example(
                text, labels, return_tensors=None, pad=pad
            )
            batch_results.append(result)

        # Combine into batch format
        if pad:
            result = {
                "input_ids": [r["input_ids"] for r in batch_results],
                "attention_mask": [r["attention_mask"] for r in batch_results],
                "label_mask": [r["label_mask"] for r in batch_results],
            }
        else:
            result = {
                "input_ids": [r["input_ids"] for r in batch_results],
                "attention_mask": [r["attention_mask"] for r in batch_results],
                "label_mask": [r["label_mask"] for r in batch_results],
            }

        if return_tensors == "pt":
            result = {
                k: torch.tensor(
                    v, dtype=torch.long if k != "label_mask" else torch.bool
                )
                for k, v in result.items()
            }

        return result

    def __call__(
        self,
        texts: Union[List[str], str],
        labels: List[Union[List[str], str]],
        return_tensors: Optional[str] = "pt",
        pad: bool = True,
    ):
        """Main tokenization method that delegates to appropriate method."""
        if isinstance(texts, str):
            return self.tokenize_example(
                texts, labels, return_tensors=return_tensors, pad=pad
            )
        return self.tokenize_batch(
            texts, labels, return_tensors=return_tensors, pad=pad
        )

    def decode_sequence(self, input_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(
            [i for i in input_ids if i != self.pad_token_id], skip_special_tokens=True
        )

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)

    # Additional methods to make it more transformers-compatible
    def batch_decode(self, sequences: Union[List[int], List[List[int]]], **kwargs):
        """Batch decode sequences."""
        return self.tokenizer.batch_decode(sequences, **kwargs)

    def decode(self, token_ids: List[int], **kwargs):
        """Decode a single sequence."""
        return self.tokenizer.decode(token_ids, **kwargs)

    @property
    def vocab_size(self) -> int:
        """Vocabulary size property."""
        return len(self.tokenizer)

    @property
    def model_max_length(self) -> int:
        """Model max length property."""
        return self.tokenizer.model_max_length

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save the tokenizer to a directory."""
        return self.tokenizer.save_pretrained(save_directory, **kwargs)


def load_dataset(
    path: str = "alexneakameni/ZSHOT-HARDSET",
    name: str = "triplet",
    split: str = "train",
    text_column: str = "sentence",
    positive_column: str = "labels",
    negative_column: str = "not_labels",
    shuffle_labels: bool = True,
    max_labels: Optional[int] = None,
):
    def mapper(x):
        labels = x[positive_column] + x[negative_column]
        labels_int = [1] * len(x[positive_column]) + [0] * len(x[negative_column])
        if shuffle_labels:
            combined = list(zip(labels, labels_int))
            random.shuffle(combined)
            labels, labels_int = zip(*combined)
        if max_labels is not None:
            labels = labels[:max_labels]
            labels_int = labels_int[:max_labels]
        return {
            "text": x[text_column],
            "labels_text": labels,
            "labels_int": labels_int,
        }

    ds = datasets.load_dataset(path, name)[split]
    ds = ds.map(
        mapper,
    )
    return ds.select_columns(["text", "labels_text", "labels_int"])


def add_tokenizer(
    dataset: datasets.Dataset,
    tokenizer: GliZNETTokenizer,
    text_column: str = "text",
    labels_text_column: str = "labels_text",
    labels_int_column: str = "labels_int",
):
    def tokenize_example(example: dict):
        results = tokenizer(example[text_column], example[labels_text_column])
        results["labels"] = (
            [
                torch.tensor(lab, dtype=torch.float32)
                for lab in example[labels_int_column]
            ]
            if isinstance(example[labels_int_column], list)
            else torch.tensor(example[labels_int_column], dtype=torch.float32)
        )
        return results

    dataset.set_transform(tokenize_example)
    return dataset
