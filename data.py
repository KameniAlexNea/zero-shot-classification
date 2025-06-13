import datasets
from gliznet.tokenizer import GliZNETTokenizer


def load_dataset(
    path: str = "alexneakameni/ZSHOT-HARDSET",
    name: str = "triplet",
    split: str = "train",
    text_column: str = "sentence",
    positive_column: str = "labels",
    negative_column: str = "not_labels",
):
    ds = datasets.load_dataset(path, name)[split]
    ds = ds.map(
        lambda x: {
            "text": x[text_column],
            "labels_text": x[positive_column] + x[negative_column],
            "labels_int": [1] * len(x[positive_column]) + [0] * len(x[negative_column]),
        },
    )
    return ds


def add_tokenizer(
    dataset: datasets.Dataset,
    tokenizer: GliZNETTokenizer,
    text_column: str = "text",
    labels_text_column: str = "labels_text",
    labels_int_column: str = "labels_int",
):
    def tokenize_example(example):
        results = tokenizer(example[text_column], example[labels_text_column])
        results["labels"] = example[labels_int_column]
        return results

    tokenized_ds = dataset.with_transform(
        lambda x: tokenizer()
    )
    return tokenized_ds