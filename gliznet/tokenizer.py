from typing import Any, Dict, List, Union

import torch
from transformers import AutoTokenizer, BertTokenizerFast

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
        *args,
        **kwargs,
    ):
        self.tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )
        self.min_text_token = min_text_token

        self.max_length = self.tokenizer.model_max_length
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

    def _build_sequence(
        self, text_tokens: List[int], label_tokens: List[List[int]]
    ) -> tuple[List[int], List[int]]:
        # Pre-calculate total sequence length to avoid multiple list extensions
        total_label_tokens = sum(len(tokens) for tokens in label_tokens)
        total_length = 1 + len(text_tokens) + 1 + total_label_tokens + len(label_tokens)

        # Pre-allocate lists with known size
        sequence = [0] * total_length
        labels_mask = [0] * total_length

        # Fill the sequence efficiently
        idx = 0
        sequence[idx] = self.cls_token_id
        idx += 1

        # Add text tokens
        sequence[idx : idx + len(text_tokens)] = text_tokens
        idx += len(text_tokens)

        # Add separator after text
        sequence[idx] = self.sep_token_id
        idx += 1

        # Add label tokens and separators
        for tokens in label_tokens:
            if tokens:  # Only process non-empty token lists
                labels_mask[idx] = 1  # Mark first token of each label
                sequence[idx : idx + len(tokens)] = tokens
                idx += len(tokens)
                sequence[idx] = self.sep_token_id
                idx += 1

        return sequence, labels_mask

    def _truncate_text_tokens(
        self, text_tokens: List[int], label_tokens: List[List[int]]
    ) -> List[int]:
        label_flat_count = sum([len(sub) for sub in label_tokens])
        # Ensure sep_count is not negative if label_tokens is empty
        sep_count = max(0, len(label_tokens) - 1)
        reserve = (
            2 + label_flat_count + sep_count
        )  # CLS + SEP_after_text + labels_tokens + SEPs_between_labels

        allowed_text = (
            self.max_length - reserve
            if reserve < self.max_length
            else self.min_text_token
        )

        # Ensure we don't go below minimum text tokens but also don't exceed what's available
        max_allowed_text = max(self.min_text_token, allowed_text)
        truncated_length = min(len(text_tokens), max_allowed_text)

        return text_tokens[:truncated_length]

    def _pad_and_mask(self, token_ids: List[int], lmask: List[bool]) -> Dict[str, Any]:
        side = getattr(self.tokenizer, "padding_side", "right")
        pad_len = self.max_length - len(token_ids)
        if side == "left":
            input_ids = [self.pad_token_id] * pad_len + token_ids
            attention_mask = [0] * pad_len + [1] * len(token_ids)
            lmask = [False] * pad_len + lmask
        else:
            input_ids = token_ids + [self.pad_token_id] * pad_len
            attention_mask = [1] * len(token_ids) + [0] * pad_len
            lmask = lmask + [False] * pad_len
        if len(lmask) > self.max_length:
            raise ValueError(
                f"Label mask length {len(lmask)} exceeds max length {self.max_length}. Please check your input."
                + self.decode_sequence(input_ids)
            )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "lmask": lmask,
        }

    def _batch_tokenize(
        self,
        texts: Union[str, List[str]],
        all_labels: Union[List[str], List[List[str]]],
    ):
        if isinstance(texts, str):
            return (
                self.tokenizer(texts, **tokenizer_config)["input_ids"],
                self.tokenizer(all_labels, **tokenizer_config)["input_ids"],
            )
        text_ids = self.tokenizer(texts, **tokenizer_config)["input_ids"]
        merged_labels = sum(all_labels, start=[])
        merged_labels_ids = self.tokenizer(merged_labels, **tokenizer_config)[
            "input_ids"
        ]
        labels_ids = []
        idx = 0
        for label_group in all_labels:
            labels_ids.append(merged_labels_ids[idx : idx + len(label_group)])
            idx += len(label_group)

        return text_ids, labels_ids

    def tokenize_example(
        self,
        text: str,
        all_labels: List[str],
    ) -> Dict[str, Any]:
        text_tokens, label_tokens = self._batch_tokenize(text, all_labels)
        text_tokens = self._truncate_text_tokens(text_tokens, label_tokens)
        token_ids, lmask = self._build_sequence(text_tokens, label_tokens)

        result = self._pad_and_mask(token_ids, lmask)
        result.update(self._to_tensors(result))
        return result

    def tokenize_batch(
        self,
        texts: List[str],
        all_labels: List[List[str]],
    ) -> Dict[str, Any]:
        text_tokens, label_tokens = self._batch_tokenize(texts, all_labels)
        text_tokens = [
            self._truncate_text_tokens(text, labels)
            for text, labels in zip(text_tokens, label_tokens)
        ]
        token_ids, label_masks = zip(
            *[
                self._build_sequence(text, labels)
                for text, labels in zip(text_tokens, label_tokens)
            ]
        )
        padded_results = [
            self._pad_and_mask(ids, mask) for ids, mask in zip(token_ids, label_masks)
        ]
        result = {
            "input_ids": [r["input_ids"] for r in padded_results],
            "attention_mask": [r["attention_mask"] for r in padded_results],
            "lmask": [r["lmask"] for r in padded_results],
        }
        result.update(self._to_tensors(result))
        return result

    def __call__(
        self,
        texts: Union[List[str], str],
        labels: List[Union[List[str], str]],
    ):
        if isinstance(texts, str):
            return self.tokenize_example(texts, labels)
        return self.tokenize_batch(texts, labels)

    def decode_sequence(self, input_ids: List[int]) -> str:
        return self.tokenizer.decode(
            [i for i in input_ids if i != self.pad_token_id], skip_special_tokens=True
        )

    def get_vocab_size(self) -> int:
        return len(self.tokenizer)

    def _to_tensors(self, results: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Convert the lists in results to torch tensors:
        input_ids, attention_mask -> long; lmask -> bool.
        """
        return {
            k: torch.tensor(
                results[k],
                dtype=torch.bool if k == "lmask" else torch.long,
            )
            for k in ("input_ids", "attention_mask", "lmask")
        }
