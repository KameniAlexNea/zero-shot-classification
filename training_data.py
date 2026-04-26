"""Project-specific dataset loaders for GliZNet training.

Each loader normalises a public HuggingFace dataset into the GliZNet format:
    - text  (str)
    - ltext (list[str]) — label strings
    - lint  (list[int]) — 1 for positive, 0 for negative

Add or remove entries from ``additional_datasets`` to control which
supplementary datasets are mixed in during training.
"""

from typing import Any, Callable, Dict, Optional

import datasets

from gliznet.training_config import LabelName

selected_columns = ["text", LabelName.ltext, LabelName.lint]


def ensure_string(value: Any) -> str:
    """Ensure value is a non-empty string."""
    if value is None or not isinstance(value, str):
        return ""
    return str(value).replace("_", " ").strip()


def validate_and_filter_dataset(ds: datasets.Dataset) -> datasets.Dataset:
    """Filter out entries with empty text or labels, ensuring all are strings."""

    def batch_filter(batch):
        texts = batch["text"]
        ltexts = batch[LabelName.ltext]
        lints = batch[LabelName.lint]

        valid_entries = []
        for text, ltext_list, lint_list in zip(texts, ltexts, lints):
            text_str = ensure_string(text)
            if not text_str:
                valid_entries.append(False)
                continue

            if not isinstance(ltext_list, list) or len(ltext_list) == 0:
                valid_entries.append(False)
                continue

            ltext_strings = [ensure_string(item) for item in ltext_list]
            if not all(ltext_strings):
                valid_entries.append(False)
                continue

            if not isinstance(lint_list, list) or len(lint_list) != len(ltext_list):
                valid_entries.append(False)
                continue

            valid_entries.append(True)

        return valid_entries

    return ds.filter(batch_filter, batched=True, batch_size=100_000)


def create_mcq_mapper(
    text_column: str,
    choices_column: str = "choices",
    answer_key_column: str = "answerKey",
    choices_text_key: str = "text",
    choices_label_key: str = "label",
) -> Callable:
    """Create a mapper function for multiple-choice question datasets."""

    def mapper(x: Dict[str, Any]) -> Dict[str, Any]:
        choices = x[choices_column]
        text = ensure_string(x[text_column])

        if isinstance(choices, dict):
            ltext = [
                ensure_string(choice) for choice in choices.get(choices_text_key, [])
            ]
            labels = choices.get(choices_label_key, [])
        else:
            ltext = [ensure_string(choice) for choice in choices]
            labels = list(range(len(choices)))

        answer_key = x[answer_key_column]
        lint = [int(i == answer_key) for i in labels]

        return {
            "text": text,
            LabelName.ltext: ltext,
            LabelName.lint: lint,
        }

    return mapper


def load_dataset_with_validation(
    ds_name: str,
    name: Optional[str] = None,
    split: str = "train",
    mapper_func: Optional[Callable] = None,
    max_size: Optional[int] = None,
    seed: int = 42,
) -> datasets.Dataset:
    """Load a HuggingFace dataset, apply an optional mapper, and validate."""
    try:
        ds = datasets.load_dataset(ds_name, name, split=split)
        if max_size is not None and len(ds) > max_size:
            ds = ds.shuffle(seed=seed).select(range(max_size))
        if mapper_func:
            ds = ds.map(mapper_func)
        ds = ds.select_columns(selected_columns)
        return validate_and_filter_dataset(ds)
    except Exception as e:
        print(f"Error loading dataset {ds_name}: {e}")
        return datasets.Dataset.from_list([])


def load_allenai_ai2_arc_easy(max_size: Optional[int] = None, seed: int = 42):
    """Load ARC-Easy dataset."""
    mapper = create_mcq_mapper("question")
    return load_dataset_with_validation(
        "allenai/ai2_arc", "ARC-Easy", mapper_func=mapper, max_size=max_size, seed=seed
    )


def load_allenai_ai2_arc_challenge(max_size: Optional[int] = None, seed: int = 42):
    """Load ARC-Challenge dataset."""
    mapper = create_mcq_mapper("question")
    return load_dataset_with_validation(
        "allenai/ai2_arc",
        "ARC-Challenge",
        mapper_func=mapper,
        max_size=max_size,
        seed=seed,
    )


def load_allenai_openbookqa(max_size: Optional[int] = None, seed: int = 42):
    """Load OpenBookQA dataset."""
    mapper = create_mcq_mapper("question_stem")
    return load_dataset_with_validation(
        "allenai/openbookqa",
        "additional",
        mapper_func=mapper,
        max_size=max_size,
        seed=seed,
    )


def load_tau_commonsense_qa(max_size: Optional[int] = None, seed: int = 42):
    """Load CommonsenseQA dataset."""
    mapper = create_mcq_mapper("question")
    return load_dataset_with_validation(
        "tau/commonsense_qa", None, mapper_func=mapper, max_size=max_size, seed=seed
    )


def load_Salesforce_cos_e(max_size: Optional[int] = None, seed: int = 42):
    """Load CoS-E dataset."""

    def mapper(x: Dict[str, Any]) -> Dict[str, Any]:
        text = ensure_string(x["question"])
        choices = x["choices"]
        ltext = [ensure_string(choice) for choice in choices]
        lint = [int(i == x["answer"]) for i in choices]

        return {
            "text": text,
            LabelName.ltext: ltext,
            LabelName.lint: lint,
        }

    return load_dataset_with_validation(
        "Salesforce/cos_e", "v1.11", mapper_func=mapper, max_size=max_size, seed=seed
    )


def load_onionmonster_dream(max_size: Optional[int] = None, seed: int = 42):
    """Load DREAM dataset."""

    def mapper_func(ds):
        raws = []
        for x in ds:
            for query in x["1"]:
                text = ensure_string("\n".join(x["0"]) + "\n\n" + query["question"])
                ltext = [ensure_string(choice) for choice in query["choice"]]
                lint = [int(i == query["answer"]) for i in query["choice"]]
                raws.append(
                    {
                        "text": text,
                        LabelName.ltext: ltext,
                        LabelName.lint: lint,
                    }
                )
        return datasets.Dataset.from_list(raws)

    ds = datasets.load_dataset("onionmonster/dream", None, split="train")
    if max_size is not None and len(ds) > max_size:
        ds = ds.shuffle(seed=seed).select(range(max_size))
    ds = mapper_func(ds)
    return validate_and_filter_dataset(ds.select_columns(selected_columns))


def _make_true_all_labels_mapper() -> Callable:
    """Shared mapper for datasets with true_labels / all_labels columns."""

    def mapper(x: Dict[str, Any]) -> Dict[str, Any]:
        ltext = [ensure_string(lab) for lab in x["all_labels"]]
        true_set = set(x["true_labels"])
        lint = [int(lab in true_set) for lab in x["all_labels"]]
        return {
            "text": ensure_string(x["text"]),
            LabelName.ltext: ltext,
            LabelName.lint: lint,
        }

    return mapper


def load_knowledgator_gliclass_v3_logic(max_size: Optional[int] = None, seed: int = 42):
    """Load knowledgator/gliclass-v3-logic-dataset."""
    return load_dataset_with_validation(
        "knowledgator/gliclass-v3-logic-dataset",
        None,
        mapper_func=_make_true_all_labels_mapper(),
        max_size=max_size,
        seed=seed,
    )


def load_biomike_formal_logic_reasoning(max_size: Optional[int] = None, seed: int = 42):
    """Load BioMike/formal-logic-reasoning-gliclass-2k."""
    return load_dataset_with_validation(
        "BioMike/formal-logic-reasoning-gliclass-2k",
        None,
        mapper_func=_make_true_all_labels_mapper(),
        max_size=max_size,
        seed=seed,
    )


# Registry of additional datasets to mix in during training.
# Comment out or remove entries to disable specific sources.
additional_datasets = {
    "allenai_ai2_arc_easy": load_allenai_ai2_arc_easy,
    "allenai_ai2_arc_challenge": load_allenai_ai2_arc_challenge,
    "allenai_openbookqa": load_allenai_openbookqa,
    "tau_commonsense_qa": load_tau_commonsense_qa,
    "Salesforce_cos_e": load_Salesforce_cos_e,
    "onionmonster_dream": load_onionmonster_dream,
    "knowledgator_gliclass_v3_logic": load_knowledgator_gliclass_v3_logic,
    "biomike_formal_logic_reasoning": load_biomike_formal_logic_reasoning,
}
