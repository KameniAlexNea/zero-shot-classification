"""Text augmentation for simulating real-world noisy text.

Introduces common human typing errors found in tweets, user-generated content,
and informal text: character drops, suffix truncation, character swaps,
missing words, etc.

Design:
    - Each augmentation is a class with __call__(text) -> text
    - AugmentationPipeline composes multiple augmentations with per-augmentation probability

Note: The `nlpaug` package (https://github.com/makcedward/nlpaug) provides similar
functionality (KeyboardAug, RandomCharAug, OcrAug, SpellingAug, RandomWordAug)
with a Flow pipeline. We keep a lightweight implementation here to avoid the heavy
dependency and maintain full control over augmentation behaviour during training.
"""

import random
import string
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import yaml


class TextAugmentation(ABC):
    """Base class for text augmentations.

    Each augmentation takes a string and returns an augmented string.
    """

    @abstractmethod
    def __call__(self, text: str) -> str:
        """Apply augmentation to text."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ─── Character-level augmentations ────────────────────────────────────────────


# Common suffix truncations (e.g., "running" -> "runnin", "artist" -> "artis")
SUFFIX_TRUNCATIONS = [
    ("ing", "in"),
    ("ist", "is"),
    ("tion", "ton"),
    ("ment", "men"),
    ("ness", "nes"),
    ("ight", "ite"),
    ("ould", "oud"),
    ("ough", "uff"),
    ("ther", "der"),
    ("ally", "aly"),
    ("ious", "ius"),
    ("eous", "eus"),
    ("ible", "able"),
    ("ance", "ence"),
]

# Adjacent keys on QWERTY keyboard
ADJACENT_KEYS = {
    "a": "sqwz",
    "b": "vghn",
    "c": "xdfv",
    "d": "sfcxer",
    "e": "wrsdf",
    "f": "dgcvrt",
    "g": "fhbvty",
    "h": "gjbnyu",
    "i": "ujkol",
    "j": "hknmui",
    "k": "jlmio",
    "l": "kop",
    "m": "njk",
    "n": "bhjm",
    "o": "iklp",
    "p": "ol",
    "q": "wa",
    "r": "edft",
    "s": "awedxz",
    "t": "rfgy",
    "u": "yhji",
    "v": "cfgb",
    "w": "qase",
    "x": "zsdc",
    "y": "tghu",
    "z": "asx",
}


class SuffixTruncation(TextAugmentation):
    """Truncate common word suffixes to simulate informal/lazy spelling.

    Examples: "running" -> "runnin", "artist" -> "artis"
    """

    def __init__(self, word_prob: float = 0.15):
        self.word_prob = word_prob

    def _truncate(self, word: str) -> str:
        for full, truncated in SUFFIX_TRUNCATIONS:
            if word.endswith(full) and len(word) > len(full) + 2:
                return word[: -len(full)] + truncated
        return word

    def __call__(self, text: str) -> str:
        words = text.split()
        result = []
        for word in words:
            if len(word) > 4 and random.random() < self.word_prob:
                if word[-1] in string.punctuation:
                    result.append(self._truncate(word[:-1]) + word[-1])
                else:
                    result.append(self._truncate(word))
            else:
                result.append(word)
        return " ".join(result)


class CharSwap(TextAugmentation):
    """Swap two adjacent characters in words.

    Example: "point" -> "poitn"
    """

    def __init__(self, word_prob: float = 0.1):
        self.word_prob = word_prob

    def _swap(self, word: str) -> str:
        if len(word) < 4:
            return word
        idx = random.randint(1, len(word) - 2)
        chars = list(word)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)

    def __call__(self, text: str) -> str:
        words = text.split()
        result = []
        for word in words:
            if len(word) > 3 and random.random() < self.word_prob:
                if word[-1] in string.punctuation:
                    result.append(self._swap(word[:-1]) + word[-1])
                else:
                    result.append(self._swap(word))
            else:
                result.append(word)
        return " ".join(result)


class CharDrop(TextAugmentation):
    """Drop a random character from words.

    Example: "running" -> "runing"
    """

    def __init__(self, word_prob: float = 0.1):
        self.word_prob = word_prob

    def _drop(self, word: str) -> str:
        if len(word) < 4:
            return word
        idx = random.randint(1, len(word) - 2)
        return word[:idx] + word[idx + 1 :]

    def __call__(self, text: str) -> str:
        words = text.split()
        result = []
        for word in words:
            if len(word) > 3 and random.random() < self.word_prob:
                if word[-1] in string.punctuation:
                    result.append(self._drop(word[:-1]) + word[-1])
                else:
                    result.append(self._drop(word))
            else:
                result.append(word)
        return " ".join(result)


class CharDuplicate(TextAugmentation):
    """Duplicate a random character in words.

    Example: "hello" -> "helllo"
    """

    def __init__(self, word_prob: float = 0.05):
        self.word_prob = word_prob

    def _duplicate(self, word: str) -> str:
        if len(word) < 3:
            return word
        idx = random.randint(0, len(word) - 1)
        return word[: idx + 1] + word[idx] + word[idx + 1 :]

    def __call__(self, text: str) -> str:
        words = text.split()
        result = []
        for word in words:
            if len(word) > 3 and random.random() < self.word_prob:
                if word[-1] in string.punctuation:
                    result.append(self._duplicate(word[:-1]) + word[-1])
                else:
                    result.append(self._duplicate(word))
            else:
                result.append(word)
        return " ".join(result)


class KeyboardTypo(TextAugmentation):
    """Replace a character with an adjacent QWERTY key.

    Simulates finger-slip typos on a physical keyboard.
    """

    def __init__(self, word_prob: float = 0.08):
        self.word_prob = word_prob

    def _typo(self, word: str) -> str:
        if len(word) < 3:
            return word
        idx = random.randint(0, len(word) - 1)
        char = word[idx].lower()
        if char in ADJACENT_KEYS:
            replacement = random.choice(ADJACENT_KEYS[char])
            if word[idx].isupper():
                replacement = replacement.upper()
            return word[:idx] + replacement + word[idx + 1 :]
        return word

    def __call__(self, text: str) -> str:
        words = text.split()
        result = []
        for word in words:
            if len(word) > 3 and random.random() < self.word_prob:
                if word[-1] in string.punctuation:
                    result.append(self._typo(word[:-1]) + word[-1])
                else:
                    result.append(self._typo(word))
            else:
                result.append(word)
        return " ".join(result)


# ─── Word-level augmentations ─────────────────────────────────────────────────


class WordDrop(TextAugmentation):
    """Drop random words from the text (simulates missing/skipped words).

    Won't drop first or last word to preserve sentence structure.
    """

    def __init__(self, word_prob: float = 0.05):
        self.word_prob = word_prob

    def __call__(self, text: str) -> str:
        words = text.split()
        if len(words) <= 3:
            return text
        result = [words[0]]
        for word in words[1:-1]:
            if random.random() >= self.word_prob:
                result.append(word)
        result.append(words[-1])
        return " ".join(result)


class RandomCaseChange(TextAugmentation):
    """Randomly change case of words (simulates informal/lazy typing).

    Can lowercase capitalized words or uppercase random words.
    """

    def __init__(self, word_prob: float = 0.05):
        self.word_prob = word_prob

    def __call__(self, text: str) -> str:
        words = text.split()
        result = []
        for word in words:
            if random.random() < self.word_prob and len(word) > 1:
                if word[0].isupper():
                    result.append(word[0].lower() + word[1:])
                else:
                    result.append(word)
            else:
                result.append(word)
        return " ".join(result)


# ─── Pipeline ─────────────────────────────────────────────────────────────────


class AugmentationPipeline(TextAugmentation):
    """Compose multiple augmentations, each applied with a given probability.

    This is the main entry point for text augmentation. It applies a list of
    augmentations sequentially, where each augmentation is applied independently
    with its own probability.

    Example:
        >>> pipeline = AugmentationPipeline([
        ...     (0.3, SuffixTruncation()),
        ...     (0.3, CharSwap()),
        ...     (0.2, CharDrop()),
        ...     (0.2, KeyboardTypo()),
        ...     (0.1, WordDrop()),
        ... ])
        >>> pipeline("The running artist is going to the meeting")
        'The runnin artis is going to the meetin'
    """

    def __init__(self, augmentations: list[tuple[float, TextAugmentation]]):
        """
        Args:
            augmentations: List of (probability, augmentation) tuples.
                Each augmentation is applied with the given probability.
        """
        self.augmentations = augmentations

    def __call__(self, text: str) -> str:
        if not text or not text.strip():
            return text
        for prob, aug in self.augmentations:
            if random.random() < prob:
                text = aug(text)
        return text

    def __repr__(self) -> str:
        items = ", ".join(f"({p}, {aug!r})" for p, aug in self.augmentations)
        return f"AugmentationPipeline([{items}])"


# ─── Label-level augmentations ────────────────────────────────────────────────


class LabelAugmentation(ABC):
    """Base class for label-level augmentations.

    Operates on (labels_text, labels_int) pairs and returns modified pairs.
    """

    @abstractmethod
    def __call__(
        self, labels_text: list[str], labels_int: list[int]
    ) -> tuple[list[str], list[int]]:
        """Apply augmentation to labels."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class LabelLimit(LabelAugmentation):
    """Limit and shuffle labels, randomly selecting a subset.

    Controls the total number of labels per sample. When shuffle_labels=True,
    randomly selects between 1 and max_labels labels to vary context size
    across training epochs.
    """

    def __init__(
        self,
        max_labels: int = 20,
        shuffle_labels: bool = True,
        remove_underscores: float = 0.9,
    ):
        self.max_labels = max_labels
        self.shuffle_labels = shuffle_labels
        self.remove_underscores = remove_underscores

    def __call__(
        self, labels_text: list[str], labels_int: list[int]
    ) -> tuple[list[str], list[int]]:
        # Replace underscores with spaces stochastically
        labels_text = [
            i.replace("_", " ") if random.random() < self.remove_underscores else i
            for i in labels_text
        ]

        combined = list(zip(labels_text, labels_int))

        if self.shuffle_labels and combined:
            random.shuffle(combined)
            num_labels = random.randint(1, min(self.max_labels, len(combined)))
            selected_pairs = combined[:num_labels]
        else:
            selected_pairs = combined[: self.max_labels]

        if not selected_pairs:
            return [], []

        labels_text, labels_int = zip(*selected_pairs)
        return list(labels_text), list(labels_int)


class NegativeRatioEnforcement(LabelAugmentation):
    """Enforce a realistic positive-to-negative ratio.

    In real-world classification (tweets, user text, etc.), typically there is
    1 positive label for every 5-10 negatives. Synthetic data often has a more
    balanced ratio. This augmentation stochastically drops positives or adds
    emphasis to negatives to simulate realistic distributions.

    Strategy:
        1. Keep at least `min_positives` positive labels
        2. Target a ratio of `neg_ratio_min` to `neg_ratio_max` negatives per positive
        3. If more positives exist than the target allows, randomly drop some
    """

    def __init__(
        self,
        min_positives: int = 1,
        max_positives: int = 3,
        neg_ratio_min: int = 3,
        neg_ratio_max: int = 10,
    ):
        self.min_positives = min_positives
        self.max_positives = max_positives
        self.neg_ratio_min = neg_ratio_min
        self.neg_ratio_max = neg_ratio_max

    def __call__(
        self, labels_text: list[str], labels_int: list[int]
    ) -> tuple[list[str], list[int]]:
        if not labels_text:
            return labels_text, labels_int

        # Separate positives and negatives
        positives = [(t, i) for t, i in zip(labels_text, labels_int) if i == 1]
        negatives = [(t, i) for t, i in zip(labels_text, labels_int) if i == 0]

        if not positives:
            # No positives — return as-is (all-negative sample)
            return labels_text, labels_int

        # Decide how many positives to keep (simulate real-world: usually 1-3)
        num_pos = random.randint(
            self.min_positives, min(self.max_positives, len(positives))
        )
        random.shuffle(positives)
        selected_positives = positives[:num_pos]

        # Target number of negatives based on ratio
        target_neg = num_pos * random.randint(self.neg_ratio_min, self.neg_ratio_max)
        if negatives:
            random.shuffle(negatives)
            selected_negatives = negatives[: min(target_neg, len(negatives))]
        else:
            selected_negatives = []

        # Combine and shuffle
        combined = selected_positives + selected_negatives
        random.shuffle(combined)

        labels_text, labels_int = zip(*combined)
        return list(labels_text), list(labels_int)


class LabelAugmentationPipeline:
    """Compose multiple label augmentations applied sequentially.

    Example:
        >>> pipeline = LabelAugmentationPipeline([
        ...     LabelLimit(max_labels=20, shuffle_labels=True),
        ...     NegativeRatioEnforcement(min_positives=1, neg_ratio_min=3, neg_ratio_max=10),
        ... ])
    """

    def __init__(self, augmentations: list[LabelAugmentation]):
        self.augmentations = augmentations

    def __call__(
        self, labels_text: list[str], labels_int: list[int]
    ) -> tuple[list[str], list[int]]:
        for aug in self.augmentations:
            labels_text, labels_int = aug(labels_text, labels_int)
        return labels_text, labels_int

    def __repr__(self) -> str:
        items = ", ".join(repr(aug) for aug in self.augmentations)
        return f"LabelAugmentationPipeline([{items}])"


# Registry of all available augmentation classes by name
AUGMENTATION_REGISTRY: dict[str, type[TextAugmentation]] = {
    "SuffixTruncation": SuffixTruncation,
    "CharSwap": CharSwap,
    "CharDrop": CharDrop,
    "CharDuplicate": CharDuplicate,
    "KeyboardTypo": KeyboardTypo,
    "WordDrop": WordDrop,
    "RandomCaseChange": RandomCaseChange,
}

LABEL_AUGMENTATION_REGISTRY: dict[str, type[LabelAugmentation]] = {
    "LabelLimit": LabelLimit,
    "NegativeRatioEnforcement": NegativeRatioEnforcement,
}


def load_augmentation_pipeline(
    config_path: Optional[str] = None,
) -> AugmentationPipeline:
    """Load an augmentation pipeline from a YAML config file.

    YAML format::

        augmentations:
          - name: SuffixTruncation
            prob: 0.4
            params:
              word_prob: 0.15
          - name: CharSwap
            prob: 0.4
            params:
              word_prob: 0.10

    Args:
        config_path: Path to YAML config. If None, uses default_augmentation_pipeline().

    Returns:
        Configured AugmentationPipeline
    """
    if config_path is None:
        return default_augmentation_pipeline()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Augmentation config not found: {config_path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    entries = config.get("augmentations", [])
    augmentations = []
    for entry in entries:
        name = entry["name"]
        prob = entry.get("prob", 0.3)
        params = entry.get("params", {})

        if name not in AUGMENTATION_REGISTRY:
            raise ValueError(
                f"Unknown augmentation '{name}'. "
                f"Available: {list(AUGMENTATION_REGISTRY.keys())}"
            )

        aug_cls = AUGMENTATION_REGISTRY[name]
        augmentations.append((prob, aug_cls(**params)))

    return AugmentationPipeline(augmentations)


def load_label_augmentation_pipeline(
    config_path: Optional[str] = None,
    max_labels: int = 20,
    shuffle_labels: bool = True,
) -> LabelAugmentationPipeline:
    """Load a label augmentation pipeline from a YAML config file.

    YAML format::

        label_augmentations:
          - name: NegativeRatioEnforcement
            params:
              min_positives: 1
              max_positives: 3
              neg_ratio_min: 3
              neg_ratio_max: 10
          - name: LabelLimit
            params:
              max_labels: 20
              shuffle_labels: true

    Args:
        config_path: Path to YAML config. If None, returns a default pipeline
            with just LabelLimit using the provided max_labels/shuffle_labels.
        max_labels: Fallback max_labels when no config is provided.
        shuffle_labels: Fallback shuffle_labels when no config is provided.

    Returns:
        Configured LabelAugmentationPipeline
    """
    if config_path is None:
        return LabelAugmentationPipeline(
            [LabelLimit(max_labels=max_labels, shuffle_labels=shuffle_labels)]
        )

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Augmentation config not found: {config_path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    entries = config.get("label_augmentations", [])
    if not entries:
        # No label_augmentations section — fall back to default
        return LabelAugmentationPipeline(
            [LabelLimit(max_labels=max_labels, shuffle_labels=shuffle_labels)]
        )

    augmentations = []
    for entry in entries:
        name = entry["name"]
        params = entry.get("params", {})

        if name not in LABEL_AUGMENTATION_REGISTRY:
            raise ValueError(
                f"Unknown label augmentation '{name}'. "
                f"Available: {list(LABEL_AUGMENTATION_REGISTRY.keys())}"
            )

        aug_cls = LABEL_AUGMENTATION_REGISTRY[name]
        augmentations.append(aug_cls(**params))

    return LabelAugmentationPipeline(augmentations)


def default_augmentation_pipeline() -> AugmentationPipeline:
    """Create the default augmentation pipeline for training.

    Returns a pipeline that simulates common real-world text noise:
    typos, missing characters, informal spelling, dropped words.
    """
    return AugmentationPipeline(
        [
            (0.4, SuffixTruncation(word_prob=0.15)),
            (0.4, CharSwap(word_prob=0.10)),
            (0.3, CharDrop(word_prob=0.10)),
            (0.2, CharDuplicate(word_prob=0.05)),
            (0.3, KeyboardTypo(word_prob=0.08)),
            (0.2, WordDrop(word_prob=0.05)),
            (0.2, RandomCaseChange(word_prob=0.05)),
        ]
    )
