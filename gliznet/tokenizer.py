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

    def _build_sequence(
        self, text_tokens: List[int], label_tokens: List[List[int]]
    ) -> tuple[List[int], List[int]]:
        # [CLS] + text + [SEP] + label1 + [LAB] + label2 + [LAB] + ...
        sequence = [self.cls_token_id] + text_tokens + [self.sep_token_id]
        lmask = [0] * (1 + len(text_tokens) + 1)

        for i, label in enumerate(label_tokens, start=1):
            lab_value = i * (self.cls_separator_token == ";")
            mask_value = i * (self.cls_separator_token != ";")
            sequence += label
            lmask += [lab_value] * len(label)

            # Add [LAB] separator after each label (except the last one)
            sequence += [self.label_sep_id]
            lmask += [mask_value]  # [LAB] token not included in label group

        return sequence, lmask

    def _truncate_text_tokens(
        self, text_tokens: List[int], label_tokens: List[List[int]]
    ) -> List[int]:
        label_flat_count = sum(len(lab) for lab in label_tokens)
        # 1 SEP after text + (len(label_tokens) - 1) LAB tokens between labels
        separator_count = 1 + max(0, len(label_tokens) - 1)
        reserve = 1 + label_flat_count + separator_count  # CLS + labels + separators

        allowed_text = (
            self.max_length - reserve
            if reserve < self.max_length
            else self.min_text_token
        )
        max_allowed_text = max(self.min_text_token, allowed_text)
        return text_tokens[: min(len(text_tokens), max_allowed_text)]

    def _pad_and_mask(self, token_ids: List[int], lmask: List[int]) -> Dict[str, Any]:
        pad_len = self.max_length - len(token_ids)
        input_ids = token_ids + [self.pad_token_id] * pad_len
        attention_mask = [1] * len(token_ids) + [0] * pad_len
        lmask = lmask + [0] * pad_len

        if len(lmask) > self.max_length:
            raise ValueError(
                f"Label mask too long: {len(lmask)} > max_length {self.max_length}"
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
        token_dropout: float = 0.0,
    ) -> Dict[str, Any]:
        text_tokens, label_tokens = self._batch_tokenize(text, all_labels)
        if token_dropout > 0.0:
            text_tokens = apply_token_dropout(
                text_tokens, token_dropout, self.mask_token_id
            )
        text_tokens = self._truncate_text_tokens(text_tokens, label_tokens)
        token_ids, lmask = self._build_sequence(text_tokens, label_tokens)
        result = self._pad_and_mask(token_ids, lmask)
        result.update(self._to_tensors(result))
        return result

    def tokenize_batch(
        self,
        texts: List[str],
        all_labels: List[List[str]],
        token_dropout: float = 0.0,
    ) -> Dict[str, Any]:
        text_tokens, label_tokens = self._batch_tokenize(texts, all_labels)
        if token_dropout > 0.0:
            text_tokens = [
                apply_token_dropout(txt, token_dropout, self.mask_token_id)
                for txt in text_tokens
            ]
        text_tokens = [
            self._truncate_text_tokens(txt, lbls)
            for txt, lbls in zip(text_tokens, label_tokens)
        ]
        token_ids, label_masks = zip(
            *[
                self._build_sequence(txt, lbls)
                for txt, lbls in zip(text_tokens, label_tokens)
            ]
        )
        padded = [
            self._pad_and_mask(ids, mask) for ids, mask in zip(token_ids, label_masks)
        ]
        result = {
            "input_ids": [p["input_ids"] for p in padded],
            "attention_mask": [p["attention_mask"] for p in padded],
            "lmask": [p["lmask"] for p in padded],
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
