"""Text augmentation for simulating real-world noisy text.

Uses `nlpaug` directly for character-level and word-level augmentations.
Only SuffixTruncation and RandomCaseChange remain as custom implementations
(no nlpaug equivalent).

Design:
    - AugmentationPipeline composes nlpaug augmenters + custom callables
      with per-augmentation probability
    - YAML config maps augmentation names to nlpaug augmenters
"""

import random
import string
from abc import ABC, abstractmethod
from pathlib import Path

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import yaml


# ─── Custom augmentations (no nlpaug equivalent) ─────────────────────────────

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


class TextAugmentation(ABC):
    """Base class for custom text augmentations."""

    @abstractmethod
    def __call__(self, text: str) -> str: ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


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


class RandomCaseChange(TextAugmentation):
    """Randomly lowercase capitalized words (simulates informal/lazy typing)."""

    def __init__(self, word_prob: float = 0.05):
        self.word_prob = word_prob

    def __call__(self, text: str) -> str:
        words = text.split()
        result = []
        for word in words:
            if random.random() < self.word_prob and len(word) > 1 and word[0].isupper():
                result.append(word[0].lower() + word[1:])
            else:
                result.append(word)
        return " ".join(result)


# ─── nlpaug wrapper ──────────────────────────────────────────────────────────


class _NlpAugWrapper(TextAugmentation):
    """Thin wrapper to give nlpaug augmenters a __call__ interface."""

    def __init__(self, aug, name: str = ""):
        self._aug = aug
        self._name = name or aug.__class__.__name__

    def __call__(self, text: str) -> str:
        result = self._aug.augment(text)
        return result[0] if isinstance(result, list) else result

    def __repr__(self) -> str:
        return f"{self._name}()"


# ─── Pipeline ─────────────────────────────────────────────────────────────────


class AugmentationPipeline(TextAugmentation):
    """Compose multiple augmentations, each applied with a given probability.

    Accepts both custom TextAugmentation instances and nlpaug augmenters
    (wrapped automatically).
    """

    def __init__(self, augmentations: list[tuple[float, TextAugmentation]]):
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


# ─── Registry ─────────────────────────────────────────────────────────────────

# Factory functions that create augmenters from keyword params.
# nlpaug augmenters are wrapped in _NlpAugWrapper for a unified __call__ API.

AUGMENTATION_REGISTRY: dict[str, callable] = {
    # Custom (no nlpaug equivalent)
    "SuffixTruncation": lambda **kw: SuffixTruncation(**kw),
    "RandomCaseChange": lambda **kw: RandomCaseChange(**kw),
    # nlpaug character-level
    "KeyboardTypo": lambda **kw: _NlpAugWrapper(nac.KeyboardAug(**kw), "KeyboardTypo"),
    "OcrTypo": lambda **kw: _NlpAugWrapper(nac.OcrAug(**kw), "OcrTypo"),
    "RandomChar": lambda **kw: _NlpAugWrapper(nac.RandomCharAug(**kw), "RandomChar"),
    # nlpaug word-level
    "RandomWord": lambda **kw: _NlpAugWrapper(naw.RandomWordAug(**kw), "RandomWord"),
    "SpellingError": lambda **kw: _NlpAugWrapper(
        naw.SpellingAug(**kw), "SpellingError"
    ),
    "WordSplit": lambda **kw: _NlpAugWrapper(naw.SplitAug(**kw), "WordSplit"),
}


# ─── Label-level augmentations ────────────────────────────────────────────────


class LabelAugmentation(ABC):
    """Base class for label-level augmentations."""

    @abstractmethod
    def __call__(
        self, labels_text: list[str], labels_int: list[int]
    ) -> tuple[list[str], list[int]]: ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class LabelLimit(LabelAugmentation):
    """Limit and shuffle labels, randomly selecting a subset."""

    def __init__(
        self,
        max_labels: int = 20,
        min_labels: int = 1,
        shuffle_labels: bool = True,
        remove_underscores: float = 0.9,
    ):
        self.max_labels = max_labels
        self.min_labels = min_labels
        self.shuffle_labels = shuffle_labels
        self.remove_underscores = remove_underscores

    def __call__(
        self, labels_text: list[str], labels_int: list[int]
    ) -> tuple[list[str], list[int]]:
        labels_text = [
            i.replace("_", " ") if random.random() < self.remove_underscores else i
            for i in labels_text
        ]

        combined = list(zip(labels_text, labels_int))

        if self.shuffle_labels and combined:
            random.shuffle(combined)
            lo = min(self.min_labels, len(combined))
            hi = min(self.max_labels, len(combined))
            num_labels = random.randint(lo, hi)
            selected_pairs = combined[:num_labels]
        else:
            selected_pairs = combined[: self.max_labels]

        if not selected_pairs:
            return [], []

        labels_text, labels_int = zip(*selected_pairs)
        return list(labels_text), list(labels_int)


class RatioEnforcement(LabelAugmentation):
    """Enforce a target positive/negative label ratio.

    Configurable for both negative-heavy (few pos, many neg) and
    positive-heavy (many pos, few neg) distributions.
    """

    def __init__(
        self,
        min_positives: int = 1,
        max_positives: int = 3,
        min_negatives: int = 3,
        max_negatives: int = 10,
    ):
        self.min_positives = min_positives
        self.max_positives = max_positives
        self.min_negatives = min_negatives
        self.max_negatives = max_negatives

    def __call__(
        self, labels_text: list[str], labels_int: list[int]
    ) -> tuple[list[str], list[int]]:
        if not labels_text:
            return labels_text, labels_int

        positives = [(t, i) for t, i in zip(labels_text, labels_int) if i == 1]
        negatives = [(t, i) for t, i in zip(labels_text, labels_int) if i == 0]

        if len(positives) < self.min_positives or len(negatives) < self.min_negatives:
            return labels_text, labels_int

        num_pos = random.randint(
            self.min_positives, min(self.max_positives, len(positives))
        )
        num_neg = random.randint(
            self.min_negatives, min(self.max_negatives, len(negatives))
        )

        random.shuffle(positives)
        random.shuffle(negatives)

        combined = positives[:num_pos] + negatives[:num_neg]
        random.shuffle(combined)

        labels_text, labels_int = zip(*combined)
        return list(labels_text), list(labels_int)


class RatioEnforcementSelector(LabelAugmentation):
    """Randomly select between negative-heavy, positive-heavy, or pass-through.

    Exposes the model to varied label distributions during training:
      - negative-heavy: typical real-world (1 pos among many negs)
      - positive-heavy: multi-label scenarios (many pos, few/no negs)
      - pass-through: keep the original distribution as-is
    """

    def __init__(
        self,
        neg_prob: float = 0.5,
        pos_prob: float = 0.2,
        neg_params: dict | None = None,
        pos_params: dict | None = None,
    ):
        self.neg_prob = neg_prob
        self.pos_prob = pos_prob
        self._neg_aug = RatioEnforcement(**(neg_params or {}))
        self._pos_aug = RatioEnforcement(**(pos_params or {}))

    def __call__(
        self, labels_text: list[str], labels_int: list[int]
    ) -> tuple[list[str], list[int]]:
        r = random.random()
        if r < self.neg_prob:
            return self._neg_aug(labels_text, labels_int)
        elif r < self.neg_prob + self.pos_prob:
            return self._pos_aug(labels_text, labels_int)
        return labels_text, labels_int


class LabelAugmentationPipeline:
    """Compose multiple label augmentations applied sequentially."""

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


LABEL_AUGMENTATION_REGISTRY: dict[str, type[LabelAugmentation]] = {
    "LabelLimit": LabelLimit,
    "RatioEnforcement": RatioEnforcement,
    "RatioEnforcementSelector": RatioEnforcementSelector,
}


def load_augmentation_pipeline(config_path: str) -> AugmentationPipeline:
    """Load an augmentation pipeline from a YAML config file.

    YAML format::

        augmentations:
          - name: KeyboardTypo
            prob: 0.3
            params:
              word_prob: 0.08
          - name: SpellingError
            prob: 0.3
            params:
              word_prob: 0.08

    Args:
        config_path: Path to YAML config.

    Returns:
        Configured AugmentationPipeline
    """
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

        factory = AUGMENTATION_REGISTRY[name]
        augmentations.append((prob, factory(**params)))

    return AugmentationPipeline(augmentations)


def load_label_augmentation_pipeline(
    config_path: str,
    max_labels: int | None = None,
) -> LabelAugmentationPipeline:
    """Load a label augmentation pipeline from a YAML config file.

    YAML format::

        label_augmentations:
          - name: RatioEnforcementSelector
            params:
              neg_prob: 0.5
              pos_prob: 0.2
              neg_params: {min_positives: 1, max_positives: 3, ...}
              pos_params: {min_positives: 2, max_positives: 6, ...}
          - name: LabelLimit
            params:
              max_labels: 20
              shuffle_labels: true

    Args:
        config_path: Path to YAML config.
        max_labels: If provided, overrides max_labels in LabelLimit config
            (useful for passing the training args value).

    Returns:
        Configured LabelAugmentationPipeline
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Augmentation config not found: {config_path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    entries = config.get("label_augmentations", [])
    if not entries:
        raise ValueError(
            f"No 'label_augmentations' section in {config_path}. "
            "Define label augmentations in the config file."
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

        # Override max_labels from args if provided
        if name == "LabelLimit" and max_labels is not None:
            params["max_labels"] = max_labels

        aug_cls = LABEL_AUGMENTATION_REGISTRY[name]
        augmentations.append(aug_cls(**params))

    return LabelAugmentationPipeline(augmentations)
