import torch
from transformers import AutoTokenizer, BertTokenizerFast
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZeroShotClassificationTokenizer:
    """
    Custom tokenizer for zero-shot classification inspired by GLiNER.
    Tokenizes input in the format: [CLS] text [SEP] lab1 [SEP] lab2 [SEP] ... labn [END]
    """

    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 512):
        """
        Initialize the tokenizer.

        Args:
            model_name: HuggingFace model name for the base tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

        # Store token IDs for convenience
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

        logger.info(f"Initialized tokenizer with vocab size: {len(self.tokenizer)}")

    def tokenize_example(
        self, text: str, all_labels: List[str], return_tensors: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Tokenize a single example for zero-shot classification.

        Format: [CLS] text [SEP] positive_lab1 [SEP] positive_lab2 [SEP] ...
                negative_lab1 [SEP] negative_lab2 [SEP] ... [END]

        Args:
            text: Input text to classify
            all_labels: Combined list of positive and negative labels
            return_tensors: Format to return tensors in ('pt' for PyTorch)

        Returns:
            Dictionary containing tokenized inputs and metadata
        """

        # Create the sequence: [CLS] text [SEP] lab1 [SEP] lab2 [SEP] ... [END]
        sequence_parts = []

        # Add CLS token
        sequence_parts.append(self.tokenizer.cls_token)

        # Add text
        text_tokens = self.tokenizer.tokenize(text)
        sequence_parts.extend(text_tokens)

        # Add SEP token after text
        sequence_parts.append(self.tokenizer.sep_token)

        # Add labels with SEP tokens between them
        for i, label in enumerate(all_labels):
            label_tokens = self.tokenizer.tokenize(label)
            sequence_parts.extend(label_tokens)
            if i < len(all_labels) - 1:
                sequence_parts.append(self.tokenizer.sep_token)

        # remove END token
        # sequence_parts.append("[END]")

        # Convert tokens to IDs, but if too long truncate text only and pad left
        # identify split between text and labels
        sep_idx = sequence_parts.index(self.tokenizer.sep_token)
        label_part = sequence_parts[sep_idx:]              # includes SEP before labels
        # compute max tokens for text (exclude CLS)
        max_text_len = self.max_length - len(label_part) - 1
        if max_text_len < 0:
            raise ValueError("max_length too small to include all labels")
        # truncate text tokens at end
        text_tokens = sequence_parts[1:sep_idx][:max_text_len]
        sequence_parts = [sequence_parts[0]] + text_tokens + label_part

        # convert to ids
        orig_ids = self.tokenizer.convert_tokens_to_ids(sequence_parts)
        # left-pad to max_length
        pad_len = self.max_length - len(orig_ids)
        input_ids = [self.pad_token_id] * pad_len + orig_ids
        attention_mask = [0] * pad_len + [1] * len(orig_ids)
        # label_mask shifted by pad_len
        label_mask = self._create_label_mask(sequence_parts, all_labels, pad_len)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_mask": label_mask,
            "text": text,
            "labels": all_labels,
            "sequence_length": min(len(sequence_parts), self.max_length),
        }

        # Convert to tensors if requested
        if return_tensors == "pt":
            result["input_ids"] = torch.tensor(result["input_ids"], dtype=torch.long)
            result["attention_mask"] = torch.tensor(
                result["attention_mask"], dtype=torch.long
            )
            result["label_mask"] = torch.tensor(result["label_mask"], dtype=torch.bool)

        return result

    def _create_label_mask(
        self, sequence_parts: List[str], labels: List[str], offset: int = 0
    ) -> List[bool]:
        """
        Create a mask to identify positions of labels in the sequence.

        Args:
            sequence_parts: List of tokens in the sequence
            labels: List of labels

        Returns:
            List of boolean values indicating label positions
        """
        mask = [False] * self.max_length
        current_pos = 0
        # skip CLS & text until first SEP
        while (
            current_pos < len(sequence_parts)
            and sequence_parts[current_pos] != self.tokenizer.sep_token
        ):
            current_pos += 1
        current_pos += 1  # skip that SEP

        # mark each label token position
        for idx, label in enumerate(labels):
            token_list = self.tokenizer.tokenize(label)
            for _ in token_list:
                pos = offset + current_pos
                if pos < self.max_length:
                    mask[pos] = True
                current_pos += 1
            # skip SEP between labels
            if idx < len(labels) - 1 and current_pos < len(sequence_parts):
                current_pos += 1

        return mask

    def tokenize_batch(
        self,
        texts: List[str],
        all_labels_list: List[List[str]],
        return_tensors: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Tokenize a batch of examples.

        Args:
            texts: List of input texts
            all_labels_list: List of label lists for each text
            return_tensors: Format to return tensors in ('pt' for PyTorch)

        Returns:
            Dictionary containing batched tokenized inputs
        """
        batch_results = []

        for text, labels in zip(texts, all_labels_list):
            batch_results.append(
                self.tokenize_example(text, labels, return_tensors=None)
            )

        # Stack results
        batched = {
            "input_ids": [r["input_ids"] for r in batch_results],
            "attention_mask": [r["attention_mask"] for r in batch_results],
            "label_mask": [r["label_mask"] for r in batch_results],
            "texts": [r["text"] for r in batch_results],
            "labels": [r["labels"] for r in batch_results],
            "sequence_lengths": [r["sequence_length"] for r in batch_results],
        }

        # Convert to tensors if requested
        if return_tensors == "pt":
            batched["input_ids"] = torch.tensor(batched["input_ids"], dtype=torch.long)
            batched["attention_mask"] = torch.tensor(
                batched["attention_mask"], dtype=torch.long
            )
            batched["label_mask"] = torch.tensor(
                batched["label_mask"], dtype=torch.bool
            )

        return batched

    def decode_sequence(self, input_ids: List[int]) -> str:
        """
        Decode a sequence of token IDs back to text.

        Args:
            input_ids: List of token IDs

        Returns:
            Decoded text string
        """
        # Remove padding tokens
        input_ids = [i for i in input_ids if i != self.pad_token_id]
        # optionally skip special tokens
        return self.tokenizer.decode(input_ids, skip_special_tokens=True)

    def get_vocab_size(self) -> int:
        """Get the vocabulary size of the tokenizer."""
        return len(self.tokenizer)


# Example usage and testing
if __name__ == "__main__":
    # Initialize tokenizer
    tokenizer = ZeroShotClassificationTokenizer()

    # Example data
    text = "A fascinating study published in the latest edition of Science Daily reveals that ancient humans used complex mathematical calculations for navigation and resource management."
    positive_labels = [
        "archaeological_findings",
        "scientific_discovery",
        "academic_publication",
    ]
    negative_labels = [
        "historical_research",
        "science_fiction_story",
        "mathematics_education",
    ]

    # Tokenize single example
    result = tokenizer.tokenize_example(
        text, positive_labels + negative_labels, return_tensors="pt"
    )

    print("Tokenization Results:")
    print(f"Input IDs shape: {result['input_ids'].shape}")
    print(f"Attention mask shape: {result['attention_mask'].shape}")
    print(f"Sequence length: {result['sequence_length']}")
    print(f"Labels: {result['labels']}")

    # Decode to verify
    decoded = tokenizer.decode_sequence(result["input_ids"].tolist())
    print(f"\nDecoded sequence: {decoded}")

    # Test batch tokenization
    texts = [text, "Another example text about machine learning algorithms."]
    labels = [positive_labels + negative_labels, ["artificial_intelligence", "computer_science", "cooking", "gardening"]]

    batch_result = tokenizer.tokenize_batch(
        texts, labels, return_tensors="pt"
    )
    print(f"\nBatch input IDs shape: {batch_result['input_ids'].shape}")
    print(f"Batch attention mask shape: {batch_result['attention_mask'].shape}")
