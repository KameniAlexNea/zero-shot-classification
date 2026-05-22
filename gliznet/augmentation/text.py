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

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw


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


class PunctuationRemoval(TextAugmentation):
    """Remove punctuation characters to simulate clean/informal text.

    Prevents the model from relying on punctuation (e.g. periods, commas)
    as spurious features for classification.
    """

    def __init__(self, punct_prob: float = 0.8, keep_apostrophe: bool = True):
        self.punct_prob = punct_prob
        self.keep_apostrophe = keep_apostrophe

    def __call__(self, text: str) -> str:
        result = []
        for ch in text:
            if ch in string.punctuation:
                if self.keep_apostrophe and ch == "'":
                    result.append(ch)
                elif random.random() >= self.punct_prob:
                    result.append(ch)
            else:
                result.append(ch)
        # Collapse multiple spaces
        return " ".join("".join(result).split())


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
    "PunctuationRemoval": lambda **kw: PunctuationRemoval(**kw),
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
