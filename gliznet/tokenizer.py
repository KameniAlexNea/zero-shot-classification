import random
import torch
import datasets
from transformers import AutoTokenizer, BertTokenizerFast
from typing import List, Dict, Any, Optional, Union


class GliZNETTokenizer:
    def __init__(
        self,
        pretrained_model_name_or_path: str = "bert-base-uncased",
        *args,
        **kwargs,
    ):
        self.tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )

        self.max_length = self.tokenizer.model_max_length
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str = "bert-base-uncased", *args, **kwargs
    ) -> "GliZNETTokenizer":
        return cls(
            pretrained_model_name_or_path=pretrained_model_name_or_path, *args, **kwargs
        )

    def _build_sequence(
        self, text_tokens: List[str], label_tokens: List[List[str]]
    ) -> tuple[List[str], List[int]]:
        sequence = [self.cls_token_id] + text_tokens + [self.sep_token_id]
        labels_mask = [0] * len(text_tokens)
        for i, tokens in enumerate(label_tokens):
            sequence += tokens
            labels_mask += [1] + [0] * (len(tokens) - 1)
            if i < len(label_tokens) - 1:
                sequence.append(self.sep_token_id)
                labels_mask.append(0)
        return sequence, labels_mask

    def _truncate_text_tokens(
        self, text_tokens: List[str], label_tokens: List[List[str]]
    ) -> List[str]:
        # Account for: CLS + SEP + sum(label tokens) + SEP between labels
        label_flat = [tok for sub in label_tokens for tok in sub]
        sep_count = len(label_tokens) - 1
        reserve = 2 + len(label_flat) + sep_count  # CLS + SEP + labels
        allowed = self.max_length - reserve
        return text_tokens[: max(0, allowed)]

    def _pad_and_mask(
        self, token_ids: List[int], label_mask: List[bool]
    ) -> Dict[str, Any]:
        side = getattr(self.tokenizer, "padding_side", "right")
        pad_len = self.max_length - len(token_ids)
        if side == "left":
            input_ids = [self.pad_token_id] * pad_len + token_ids
            attention_mask = [0] * pad_len + [1] * len(token_ids)
            label_mask = [False] * pad_len + label_mask
        else:
            input_ids = token_ids + [self.pad_token_id] * pad_len
            attention_mask = [1] * len(token_ids) + [0] * pad_len
            label_mask = label_mask + [False] * pad_len
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_mask": label_mask,
        }

    def _create_label_mask(
        self, sequence: List[str], label_tokens: List[List[str]]
    ) -> List[bool]:
        mask = [False] * len(sequence)
        if not label_tokens:
            return mask
        start = sequence.index(self.sep_token_id) + 1
        idx = start
        for tokens in label_tokens:
            mask[idx] = True  # Mark the first token of the label
            idx += len(tokens) + 1  # +1 for the SEP token
        return mask

    def _batch_tokenize(
        self,
        texts: Union[str, List[str]],
        all_labels: Union[List[str], List[List[str]]],
    ):
        if isinstance(texts, str):
            return (
                self.tokenizer(
                    texts, return_attention_mask=False, return_token_type_ids=False
                )["input_ids"],
                self.tokenizer(
                    all_labels,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )["input_ids"],
            )
        return self.tokenizer(
            texts, return_attention_mask=False, return_token_type_ids=False
        )["input_ids"], [
            self.tokenizer(
                labels, return_attention_mask=False, return_token_type_ids=False
            )["input_ids"]
            for labels in all_labels
        ]

    def tokenize_example(
        self,
        text: str,
        all_labels: List[str],
        return_tensors: Optional[str] = "pt",
        pad: bool = True,
    ) -> Dict[str, Any]:
        text_tokens, label_tokens = self._batch_tokenize(text, all_labels)
        text_tokens = self._truncate_text_tokens(text_tokens, label_tokens)
        token_ids, label_mask = self._build_sequence(text_tokens, label_tokens)

        if pad:
            result = self._pad_and_mask(token_ids, label_mask)
        else:
            attention_mask = [1] * len(token_ids)
            result = {
                "input_ids": token_ids,
                "attention_mask": attention_mask,
                "label_mask": label_mask,
            }

        if return_tensors == "pt":
            # Convert to PyTorch tensors: unsqueeze to add batch dimension
            result_pad = {
                k: torch.tensor(
                    v, dtype=torch.long if k != "label_mask" else torch.bool
                )
                for k, v in result.items()
                if k in {"input_ids", "attention_mask", "label_mask"}
            }
            result.update(result_pad)
        return result

    def tokenize_batch(
        self,
        texts: List[str],
        all_labels: List[List[str]],
        return_tensors: Optional[str] = "pt",
        pad: bool = True,
    ) -> Dict[str, Any]:
        if return_tensors == "pt" and not pad:
            raise ValueError(
                "return_tensors='pt' requires pad=True for batch tokenization."
            )
        text_tokens, label_tokens = self._batch_tokenize(texts, all_labels)
        token_ids, label_masks = zip(
            *[
                self._build_sequence(text, labels)
                for text, labels in zip(text_tokens, label_tokens)
            ]
        )
        if pad:
            padded_results = [
                self._pad_and_mask(ids, mask)
                for ids, mask in zip(token_ids, label_masks)
            ]
            result = {
                "input_ids": [r["input_ids"] for r in padded_results],
                "attention_mask": [r["attention_mask"] for r in padded_results],
                "label_mask": [r["label_mask"] for r in padded_results],
            }
        else:
            result = {
                "input_ids": token_ids,
                "attention_mask": [[1] * len(ids) for ids in token_ids],
                "label_mask": label_masks,
            }
        if return_tensors == "pt":
            result_pad = {
                k: torch.tensor(
                    v, dtype=torch.long if k != "label_mask" else torch.bool
                )
                for k, v in result.items()
                if k in {"input_ids", "attention_mask", "label_mask"}
            }
            result.update(result_pad)
        return result

    def __call__(
        self,
        texts: Union[List[str], str],
        labels: List[Union[List[str], str]],
        return_tensors: Optional[str] = "pt",
        pad: bool = True,
    ):
        if isinstance(texts, str):
            return self.tokenize_example(
                texts, labels, return_tensors=return_tensors, pad=pad
            )
        return self.tokenize_batch(
            texts, labels, return_tensors=return_tensors, pad=pad
        )

    def decode_sequence(self, input_ids: List[int]) -> str:
        return self.tokenizer.decode(
            [i for i in input_ids if i != self.pad_token_id], skip_special_tokens=True
        )

    def get_vocab_size(self) -> int:
        return len(self.tokenizer)


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
