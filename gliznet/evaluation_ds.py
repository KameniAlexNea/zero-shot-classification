import json
import re

import datasets
from loguru import logger

from . import LabelName


def split_by_uppercase(text):
    return re.sub(r"(?<=[a-z])(?=[A-Z])", "_", text)


def load_dbpedia_dataset():
    test_ds = datasets.load_dataset("fancyzhx/dbpedia_14")["test"]
    all_labels = test_ds.features["label"].names
    ds_mapping = {i: split_by_uppercase(i).lower() for i in all_labels}

    def convert_labels(label: str):
        return {
            LabelName.ltext: [ds_mapping.values()],
            LabelName.lint: [label == i for i in ds_mapping],
        }

    def format_text(title: str, content: str):
        return f"{title}\n{content}"

    test_ds = test_ds.map(
        lambda x: {
            "text": format_text(
                x["title"],
                x["content"],
            ),
            **convert_labels(x["label"]),
        },
        remove_columns=test_ds.column_names,
    )
    return test_ds


def load_events_classification_biotech():
    test_ds = datasets.load_dataset(
        "knowledgator/events_classification_biotech", trust_remote_code=True
    )["test"]

    def clean_label(label: str):
        return label.lower().replace(" ", "_").replace("-", "and").replace("&", "and")

    def format_text(title: str, content: str, target_organism: str):
        return f"{title}\n{content}\nTarget Organism: {target_organism}"

    test_labels = sum(test_ds["all_labels"], start=[])
    test_labels = list(set(test_labels))
    mapping = {i: clean_label(i) for i in test_labels}

    def convert_labels(labels: list[str]):
        return {
            LabelName.ltext: list(mapping.values()),
            LabelName.lint: [i in labels for i in mapping],
        }

    test_ds = test_ds.map(
        lambda x: {
            "text": format_text(
                x["title"],
                x["content"],
                x["target organization"],
            ),
            **convert_labels(set(x["all_labels"])),
        },
        remove_columns=test_ds.column_names,
    )
    return test_ds


def load_agnews_dataset():
    test_ds = datasets.load_dataset("sh0416/ag_news")["test"]
    mapping = {1: "World", 2: "Sports", 3: "Business", 4: "Science_or_Technology"}
    mapping = {k: v.lower() + "_news" for k, v in mapping.items()}

    def convert_labels(label: int):
        return {
            LabelName.ltext: list(mapping.values()),
            LabelName.lint: [i == label for i in mapping],
        }

    test_ds = test_ds.map(
        lambda x: {
            "text": (x["title"] + "\n" + x["description"]),
            **convert_labels(x["label"]),
        },
        remove_columns=test_ds.column_names,
    )
    return test_ds


def load_imdb_dataset():
    test_ds = datasets.load_dataset("stanfordnlp/imdb")["test"]
    mapping = {0: "negative", 1: "positive"}
    mapping = {k: v.lower() + "_sentiment" for k, v in mapping.items()}

    def convert_labels(label: int):
        return {
            LabelName.ltext: list(mapping.values()),
            LabelName.lint: [i == label for i in mapping],
        }

    test_ds = test_ds.map(
        lambda x: {
            "text": x["text"],
            **convert_labels(x["label"]),
        },
        remove_columns=test_ds.column_names,
    )
    return test_ds


def load_amazon_massive_intent(grouped: bool = True):
    intent_groups: dict[str, list[str]] = json.load(
        open("gliznet/eval_data/intent_data.json", "r")
    )
    test_ds = datasets.load_dataset("mteb/amazon_massive_intent", "en")["test"]
    all_labels: list[str] = list(set(test_ds["label"]))
    mapping = (
        {
            i: intent_group + "_intent"
            for intent_group, intents in intent_groups.items()
            for i in intents
        }
        if grouped
        else {i: i for i in all_labels}
    )

    labels = list(set(mapping.values()))

    def convert_labels(label: str):
        return {
            LabelName.ltext: [i for i in labels],
            LabelName.lint: [i == mapping[label] for i in labels],
        }

    test_ds = test_ds.map(
        lambda x: {
            "text": x["text"],
            **convert_labels(x["label"]),
        },
        remove_columns=test_ds.column_names,
    )
    logger.info(f"Loaded {len(test_ds)} samples with {len(labels)} labels.")
    logger.info(f"Labels: {labels}")
    logger.info(str(test_ds[0]))
    return test_ds


ds_mapping = {
    "agnews": load_agnews_dataset,
    "imdb": load_imdb_dataset,
    "amazon_massive_intent": load_amazon_massive_intent,
}
