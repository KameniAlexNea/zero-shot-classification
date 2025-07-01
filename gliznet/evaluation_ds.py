import json

import datasets
from loguru import logger

from . import LabelName


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
        open("gliznet/intent_data.json", "r")
    )
    test_ds = datasets.load_dataset("mteb/amazon_massive_intent", "en")["test"]
    all_labels: list[str] = list(set(test_ds["label"]))
    mapping = (
        {
            i: intent_group
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
