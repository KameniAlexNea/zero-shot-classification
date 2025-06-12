import torch
from transformers import AutoTokenizer, BertTokenizerFast
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.pad_token_id = self.tokenizer.pad_token_id

        logger.info(f"Initialized tokenizer with vocab size: {len(self.tokenizer)}")

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str = "bert-base-uncased", *args, **kwargs
    ) -> "GliZNETTokenizer":
        return cls(
            pretrained_model_name_or_path=pretrained_model_name_or_path, *args, **kwargs
        )

    def _build_sequence(
        self, text_tokens: List[str], label_tokens: List[List[str]]
    ) -> List[str]:
        sequence = [self.cls_token] + text_tokens + [self.sep_token]
        for i, tokens in enumerate(label_tokens):
            sequence += tokens
            if i < len(label_tokens) - 1:
                sequence.append(self.sep_token)
        return sequence

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
        pad_len = self.max_length - len(token_ids)
        input_ids = [self.pad_token_id] * pad_len + token_ids
        attention_mask = [0] * pad_len + [1] * len(token_ids)
        label_mask = [False] * pad_len + label_mask
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_mask": label_mask,
        }

    def _create_label_mask(
        self, sequence: List[str], label_tokens: List[List[str]]
    ) -> List[bool]:
        mask = [False] * len(sequence)
        try:
            start = sequence.index(self.sep_token) + 1
        except ValueError:
            return mask

        idx = start
        for tokens in label_tokens:
            for _ in tokens:
                if idx < len(sequence):
                    mask[idx] = True
                idx += 1
            if idx < len(sequence) and sequence[idx] == self.sep_token:
                idx += 1
        return mask

    def tokenize_example(
        self, text: str, all_labels: List[str], return_tensors: Optional[str] = None
    ) -> Dict[str, Any]:
        text_tokens = self.tokenizer.tokenize(text)
        label_tokens = [self.tokenizer.tokenize(label) for label in all_labels]
        text_tokens = self._truncate_text_tokens(text_tokens, label_tokens)
        sequence = self._build_sequence(text_tokens, label_tokens)

        token_ids = self.tokenizer.convert_tokens_to_ids(sequence)
        label_mask = self._create_label_mask(sequence, label_tokens)
        padded = self._pad_and_mask(token_ids, label_mask)

        result = {
            **padded,
            "text": text,
            "labels": all_labels,
            "sequence_length": min(len(sequence), self.max_length),
        }

        if return_tensors == "pt":
            result = {
                k: torch.tensor(
                    v, dtype=torch.long if k != "label_mask" else torch.bool
                )
                for k, v in result.items()
                if k in {"input_ids", "attention_mask", "label_mask"}
            } | {
                k: v
                for k, v in result.items()
                if k not in {"input_ids", "attention_mask", "label_mask"}
            }

        return result

    def tokenize_batch(
        self,
        texts: List[str],
        all_labels_list: List[List[str]],
        return_tensors: Optional[str] = None,
    ) -> Dict[str, Any]:
        batch = []
        for text, labels in zip(texts, all_labels_list):
            ex = self.tokenize_example(text, labels)
            batch.append(ex)

        input_ids = [ex["input_ids"] for ex in batch]
        attention_mask = [ex["attention_mask"] for ex in batch]
        label_mask = [ex["label_mask"] for ex in batch]
        texts_out = [ex["text"] for ex in batch]
        labels_out = [ex["labels"] for ex in batch]
        lengths = [ex["sequence_length"] for ex in batch]

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_mask": label_mask,
            "texts": texts_out,
            "labels": labels_out,
            "sequence_lengths": lengths,
        }

        if return_tensors == "pt":
            result["input_ids"] = torch.tensor(input_ids, dtype=torch.long)
            result["attention_mask"] = torch.tensor(attention_mask, dtype=torch.long)
            result["label_mask"] = torch.tensor(label_mask, dtype=torch.bool)

        return result

    def decode_sequence(self, input_ids: List[int]) -> str:
        return self.tokenizer.decode(
            [i for i in input_ids if i != self.pad_token_id], skip_special_tokens=True
        )

    def get_vocab_size(self) -> int:
        return len(self.tokenizer)
