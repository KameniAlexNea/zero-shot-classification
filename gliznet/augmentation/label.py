"""Label-level augmentations for GliZNet training.

Controls the composition and ratio of positive/negative labels per sample,
aligning training distributions with real-world inference patterns.
"""

import random
from abc import ABC, abstractmethod


class LabelAugmentation(ABC):
    """Base class for label-level augmentations."""

    @abstractmethod
    def __call__(
        self, labels_text: list[str], labels_int: list[int]
    ) -> tuple[list[str], list[int]]: ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class LabelLimit(LabelAugmentation):
    """Limit and shuffle labels with stratified sampling.

    When ``preserve_ratio=True`` (the default), truncation uses stratified
    sampling: positives and negatives are sampled proportionally so that
    an upstream ratio-enforcement step isn't destroyed by random truncation.
    """

    def __init__(
        self,
        max_labels: int = 20,
        min_labels: int = 1,
        shuffle_labels: bool = True,
        remove_underscores: float = 0.9,
        preserve_ratio: bool = True,
    ):
        self.max_labels = max_labels
        self.min_labels = min_labels
        self.shuffle_labels = shuffle_labels
        self.remove_underscores = remove_underscores
        self.preserve_ratio = preserve_ratio

    def _stratified_select(
        self, combined: list[tuple[str, int]], num_labels: int
    ) -> list[tuple[str, int]]:
        """Select *num_labels* items while preserving the pos/neg ratio."""
        positives = [p for p in combined if p[1] == 1]
        negatives = [p for p in combined if p[1] == 0]

        n_pos = len(positives)
        n_neg = len(negatives)
        total = n_pos + n_neg

        if total == 0 or num_labels >= total:
            return combined

        # All-same-class: just truncate
        if n_pos == 0 or n_neg == 0:
            random.shuffle(combined)
            return combined[:num_labels]

        # Proportional allocation (at least 1 of each class)
        target_pos = max(1, round(num_labels * n_pos / total))
        target_neg = num_labels - target_pos
        # Clamp to available
        target_pos = min(target_pos, n_pos)
        target_neg = min(target_neg, n_neg)
        # Redistribute remainder
        remainder = num_labels - target_pos - target_neg
        if remainder > 0:
            if n_pos - target_pos > 0:
                extra_pos = min(remainder, n_pos - target_pos)
                target_pos += extra_pos
                remainder -= extra_pos
            if remainder > 0:
                target_neg += min(remainder, n_neg - target_neg)

        random.shuffle(positives)
        random.shuffle(negatives)
        selected = positives[:target_pos] + negatives[:target_neg]
        random.shuffle(selected)
        return selected

    def __call__(
        self, labels_text: list[str], labels_int: list[int]
    ) -> tuple[list[str], list[int]]:
        labels_text = [
            i.replace("_", " ") if random.random() < self.remove_underscores else i
            for i in labels_text
        ]

        combined = list(zip(labels_text, labels_int))

        if self.shuffle_labels and combined:
            lo = min(self.min_labels, len(combined))
            hi = min(self.max_labels, len(combined))
            # Triangular distribution with mode=hi: biases toward more labels
            num_labels = round(random.triangular(lo, hi, hi))

            if self.preserve_ratio:
                selected_pairs = self._stratified_select(combined, num_labels)
            else:
                random.shuffle(combined)
                selected_pairs = combined[:num_labels]
        else:
            selected_pairs = combined[: self.max_labels]

        if not selected_pairs:
            return [], []

        labels_text, labels_int = zip(*selected_pairs)
        return list(labels_text), list(labels_int)

class ScenarioAwareSampler(LabelAugmentation):
    """Scenario-aware label sampling aligned with real-world inference patterns.

    The key insight: at inference time, zero-shot classification is almost always
    "pick 1 correct label from K candidates" (needle-in-haystack), but the
    training data is dominated by multi-positive samples. This mismatch hurts
    performance on standard benchmarks.

    Scenarios:
        - **needle**: Exactly 1 positive among many negatives. Directly mimics
          single-label ZSC (the dominant inference pattern for benchmarks like
          20_newsgroups, ag_news, emotion, SST-5).
        - **few_pos**: Pick n negatives (min 4), then pick n_pos from [0, n_neg].
          Negatives dominate or equal positives. Naturally includes all-negative
          (n_pos=0) and needle-like (n_pos=1) cases.
        - **few_neg**: Pick n positives (min 4), then pick n_neg from [0, n_pos].
          Positives dominate or equal negatives. Naturally includes all-positive
          (n_neg=0) cases.
        - **balanced**: ~50/50 split. Tests fine-grained discrimination.
        - **passthrough**: Original distribution from the dataset.

    Loss interaction:
        - SoftmaxLoss returns 0 for all-negative and all-positive samples
          (by design — no contrastive signal without both classes).
        - FocalLoss provides signal for ALL scenarios.
        - The explicit scenario exposure ensures FocalLoss learns correct
          calibration for edge cases, while SoftmaxLoss focuses on the
          mixed-class scenarios where it excels.
    """

    def __init__(
        self,
        needle_prob: float = 0.20,
        few_pos_prob: float = 0.10,
        few_neg_prob: float = 0.10,
        balanced_prob: float = 0.10,
        # needle scenario
        needle_min_neg: int = 4,
        needle_max_neg: int = 15,
        # few_pos scenario: pick n_neg first, then n_pos in [0, n_neg]
        few_pos_min_neg: int = 4,
        few_pos_max_neg: int = 15,
        # few_neg scenario: pick n_pos first, then n_neg in [0, n_pos]
        few_neg_min_pos: int = 4,
        few_neg_max_pos: int = 15,
        # balanced scenario
        balanced_min_per_class: int = 2,
        balanced_max_per_class: int = 8,
    ):
        self.needle_prob = needle_prob
        self.few_pos_prob = few_pos_prob
        self.few_neg_prob = few_neg_prob
        self.balanced_prob = balanced_prob

        self.needle_min_neg = needle_min_neg
        self.needle_max_neg = needle_max_neg

        self.few_pos_min_neg = few_pos_min_neg
        self.few_pos_max_neg = few_pos_max_neg

        self.few_neg_min_pos = few_neg_min_pos
        self.few_neg_max_pos = few_neg_max_pos

        self.balanced_min_per_class = balanced_min_per_class
        self.balanced_max_per_class = balanced_max_per_class

    @staticmethod
    def _split(
        labels_text: list[str], labels_int: list[int]
    ) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
        positives = [(t, i) for t, i in zip(labels_text, labels_int) if i == 1]
        negatives = [(t, i) for t, i in zip(labels_text, labels_int) if i == 0]
        return positives, negatives

    @staticmethod
    def _merge(
        selected: list[tuple[str, int]],
    ) -> tuple[list[str], list[int]]:
        if not selected:
            return [], []
        random.shuffle(selected)
        texts, ints = zip(*selected)
        return list(texts), list(ints)

    def _needle(self, positives, negatives):
        """Exactly 1 positive among many negatives."""
        if not positives or len(negatives) < self.needle_min_neg:
            return None
        random.shuffle(positives)
        random.shuffle(negatives)
        n_neg = random.randint(
            self.needle_min_neg, min(self.needle_max_neg, len(negatives))
        )
        return positives[:1] + negatives[:n_neg]

    def _few_pos(self, positives, negatives):
        """Negatives dominate: pick n_neg first, then n_pos in [0, n_neg].

        Naturally includes all-negative (n_pos=0) and needle-like (n_pos=1).
        """
        if len(negatives) < self.few_pos_min_neg:
            return None
        random.shuffle(positives)
        random.shuffle(negatives)
        n_neg = random.randint(
            self.few_pos_min_neg, min(self.few_pos_max_neg, len(negatives))
        )
        n_pos = random.randint(0, min(n_neg, len(positives)))
        return positives[:n_pos] + negatives[:n_neg]

    def _few_neg(self, positives, negatives):
        """Positives dominate: pick n_pos first, then n_neg in [0, n_pos].

        Naturally includes all-positive (n_neg=0).
        """
        if len(positives) < self.few_neg_min_pos:
            return None
        random.shuffle(positives)
        random.shuffle(negatives)
        n_pos = random.randint(
            self.few_neg_min_pos, min(self.few_neg_max_pos, len(positives))
        )
        n_neg = random.randint(0, min(n_pos, len(negatives)))
        return positives[:n_pos] + negatives[:n_neg]

    def _balanced(self, positives, negatives):
        """Roughly equal positives and negatives."""
        if not positives or not negatives:
            return None
        random.shuffle(positives)
        random.shuffle(negatives)
        max_per = min(
            self.balanced_max_per_class, len(positives), len(negatives)
        )
        if max_per < self.balanced_min_per_class:
            return None
        n = random.randint(self.balanced_min_per_class, max_per)
        return positives[:n] + negatives[:n]

    def __call__(
        self, labels_text: list[str], labels_int: list[int]
    ) -> tuple[list[str], list[int]]:
        if not labels_text:
            return labels_text, labels_int

        positives, negatives = self._split(labels_text, labels_int)

        # Weighted random scenario selection with fallback to passthrough
        r = random.random()
        cum = 0.0
        scenarios = [
            (self.needle_prob, lambda: self._needle(positives, negatives)),
            (self.few_pos_prob, lambda: self._few_pos(positives, negatives)),
            (self.few_neg_prob, lambda: self._few_neg(positives, negatives)),
            (self.balanced_prob, lambda: self._balanced(positives, negatives)),
        ]

        for prob, fn in scenarios:
            cum += prob
            if r < cum:
                result = fn()
                if result is not None:
                    return self._merge(result)
                # Scenario not feasible (not enough labels) → passthrough
                return labels_text, labels_int

        # Passthrough: keep original distribution
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
    "ScenarioAwareSampler": ScenarioAwareSampler,
}
