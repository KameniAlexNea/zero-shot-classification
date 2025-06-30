import datasets

from .config import LabelName


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


def load_amazon_massive_intent():
    test_ds = datasets.load_dataset("mteb/amazon_massive_intent", "en")["test"]
    all_labels = list(set(test_ds["label"]))
    mapping = {label: f"intent_{i}" for i, label in enumerate(all_labels)}

    def convert_labels(label: str):
        return {
            LabelName.ltext: list(mapping.values()),
            LabelName.lint: [i == mapping[label] for i in mapping.values()],
        }

    test_ds = test_ds.map(
        lambda x: {
            "text": x["text"],
            **convert_labels(x["label"]),
        },
        remove_columns=test_ds.column_names,
    )
    return test_ds
