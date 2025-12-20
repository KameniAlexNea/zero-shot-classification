import random
from typing import Any, Callable, Dict, Optional

import datasets

from . import LabelName

selected_columns = ["text", LabelName.ltext, LabelName.lint]


def ensure_string(value: Any) -> str:
    """Ensure value is a string and not empty."""
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
            # Ensure text is a non-empty string
            text_str = ensure_string(text)
            if not text_str:
                valid_entries.append(False)
                continue

            # Ensure ltext is a list of non-empty strings
            if not isinstance(ltext_list, list) or len(ltext_list) == 0:
                valid_entries.append(False)
                continue

            ltext_strings = [ensure_string(item) for item in ltext_list]
            if not all(ltext_strings):
                valid_entries.append(False)
                continue

            # Ensure lint is a list of same length as ltext
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
    """Create a mapper function for multiple choice question datasets."""

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
) -> datasets.Dataset:
    """Load and validate a dataset with proper error handling."""
    try:
        ds = datasets.load_dataset(ds_name, name, split=split)
        if mapper_func:
            ds = ds.map(mapper_func)
        ds = ds.select_columns(selected_columns)
        return validate_and_filter_dataset(ds)
    except Exception as e:
        print(f"Error loading dataset {ds_name}: {e}")
        return datasets.Dataset.from_list([])


def load_allenai_ai2_arc_easy():
    """Load ARC-Easy dataset."""
    mapper = create_mcq_mapper("question")
    return load_dataset_with_validation(
        "allenai/ai2_arc", "ARC-Easy", mapper_func=mapper
    )


def load_allenai_ai2_arc_challenge():
    """Load ARC-Challenge dataset."""
    mapper = create_mcq_mapper("question")
    return load_dataset_with_validation(
        "allenai/ai2_arc", "ARC-Challenge", mapper_func=mapper
    )


def load_allenai_openbookqa():
    """Load OpenBookQA dataset."""
    mapper = create_mcq_mapper("question_stem")
    return load_dataset_with_validation(
        "allenai/openbookqa", "additional", mapper_func=mapper
    )


def load_tau_commonsense_qa():
    """Load CommonsenseQA dataset."""
    mapper = create_mcq_mapper("question")
    return load_dataset_with_validation("tau/commonsense_qa", None, mapper_func=mapper)


def load_Salesforce_cos_e():
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

    return load_dataset_with_validation("Salesforce/cos_e", "v1.11", mapper_func=mapper)


def load_onionmonster_dream():
    """Load DREAM dataset."""

    def mapper_func(ds):
        raws = []
        for x in ds:
            for query in x["1"]:
                text = ensure_string(f"{query['question']}\n" + "\n".join(x["0"]))
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
    ds = mapper_func(ds)
    return validate_and_filter_dataset(ds.select_columns(selected_columns))


def load_sagnikrayc_mctest():
    """Load MCTest dataset."""

    def mapper(x: Dict[str, Any]) -> Dict[str, Any]:
        text = ensure_string(f"{x['question']}\n{x['story']}")
        options = [ensure_string(str(i)) for i in x["answer_options"].values()]
        lint = [int(i == x["answer"]) for i in x["answer_options"]]

        return {
            "text": text,
            LabelName.ltext: options,
            LabelName.lint: lint,
        }

    return load_dataset_with_validation(
        "sagnikrayc/mctest", "mc500", mapper_func=mapper
    )


def load_ehovy_race():
    """Load RACE dataset."""

    def mapper(x: Dict[str, Any]) -> Dict[str, Any]:
        text = ensure_string(f"{x['question']}\n{x['article']}")
        ltext = [ensure_string(option) for option in x["options"]]
        lint = [
            int(i == (ord(x["answer"]) - ord("A"))) for i in range(len(x["options"]))
        ]

        return {
            "text": text,
            LabelName.ltext: ltext,
            LabelName.lint: lint,
        }

    return load_dataset_with_validation("ehovy/race", "all", mapper_func=mapper)


def load_sentence_transformers_wikihow():
    """Load WikiHow dataset."""

    def mapper_func(ds):
        all_labels = list(
            set(ensure_string(summary) for summary in ds["summary"] if summary)
        )

        def mapper(x: Dict[str, Any]) -> Dict[str, Any]:
            text = ensure_string(x["text"])
            summary = ensure_string(x["summary"])

            neg_count = random.randint(1, 4)
            neg_labels = random.sample(all_labels, min(neg_count, len(all_labels)))
            labels = [summary] + neg_labels
            random.shuffle(labels)

            return {
                "text": text,
                LabelName.ltext: labels,
                LabelName.lint: [int(i == summary) for i in labels],
            }

        return ds.map(mapper)

    ds = datasets.load_dataset("sentence-transformers/wikihow", None, split="train")
    ds = mapper_func(ds)
    return validate_and_filter_dataset(ds.select_columns(selected_columns))


def load_tasksource_cycic_classification():
    """Load CYCIC classification dataset."""

    def mapper(x: Dict[str, Any]) -> Dict[str, Any]:
        text = ensure_string(x["question"])
        ltext = ["false", "true"]
        lint = [1 - int(x["correct_answer"]), int(x["correct_answer"])]

        return {
            "text": text,
            LabelName.ltext: ltext,
            LabelName.lint: lint,
        }

    return load_dataset_with_validation(
        "tasksource/cycic_classification", None, mapper_func=mapper
    )


def load_ml4pubmed_pubmed_text_classification_cased():
    """Load PubMed text classification dataset."""

    def mapper_func(ds):
        labels = list(set(ensure_string(target) for target in ds["target"] if target))

        def mapper(x: Dict[str, Any]) -> Dict[str, Any]:
            text = ensure_string(x["description_cln"])
            target = ensure_string(x["target"])

            return {
                "text": text,
                LabelName.ltext: labels,
                LabelName.lint: [int(target == label) for label in labels],
            }

        return ds.map(mapper)

    ds = datasets.load_dataset(
        "ml4pubmed/pubmed-text-classification-cased", None, split="train"
    )
    ds = mapper_func(ds)
    return validate_and_filter_dataset(ds.select_columns(selected_columns))


def load_alexneakameni_qa_africa():
    """Load QA Africa dataset."""

    def mapper(x: Dict[str, Any]) -> Dict[str, Any]:
        text = ensure_string(f"{x['question_text']}\n{x['explanation']}")
        answer_choices = {
            k: ensure_string(v) for k, v in x["answer_choices"].items() if v
        }
        ltext = list(answer_choices.values())
        lint = [int(i in x["correct_answers"]) for i in answer_choices]

        return {
            "text": text,
            LabelName.ltext: ltext,
            LabelName.lint: lint,
        }

    return load_dataset_with_validation(
        "alexneakameni/qa_africa", None, mapper_func=mapper
    )


additional_datasets = {
    "allenai_ai2_arc_easy": load_allenai_ai2_arc_easy,
    "allenai_ai2_arc_challenge": load_allenai_ai2_arc_challenge,
    "allenai_openbookqa": load_allenai_openbookqa,
    "tau_commonsense_qa": load_tau_commonsense_qa,
    "Salesforce_cos_e": load_Salesforce_cos_e,
    "onionmonster_dream": load_onionmonster_dream,
    "sagnikrayc_mctest": load_sagnikrayc_mctest,
    "ehovy_race": load_ehovy_race,
    # "sentence_transformers_wikihow": load_sentence_transformers_wikihow,
    "tasksource_cycic_classification": load_tasksource_cycic_classification,
    "ml4pubmed_pubmed": load_ml4pubmed_pubmed_text_classification_cased,
    "alexneakameni_qa_africa": load_alexneakameni_qa_africa,
}
