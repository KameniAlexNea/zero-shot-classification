import datasets


class LabelName:
    ltext = "ltext"
    lint = "lint"


def load_toxic_conversations_dataset(split="test"):
    test_ds = datasets.load_dataset("mteb/toxic_conversations", split=split)
    columns = [
        "not_toxic",
        "toxic",
    ]
    test_ds = test_ds.map(
        lambda x: {
            "text": x["text"],
            LabelName.ltext: columns,
            LabelName.lint: [1 - x["label"], x["label"]],
        },
        remove_columns=test_ds.column_names,
    )
    return test_ds


def load_poem_sentiment_dataset(split="test"):
    test_ds = datasets.load_dataset("mteb/poem_sentiment", split=split)
    columns = test_ds.features["label"].names
    test_ds = test_ds.map(
        lambda x: {
            "text": x["text"],
            LabelName.ltext: columns,
            LabelName.lint: [x["label"] == i for i in range(len(columns))],
        },
        remove_columns=test_ds.column_names,
    )
    return test_ds


def load_movie_review_sentiment_dataset(split="test"):
    test_ds = datasets.load_dataset("mteb/movie_review_sentiment", split=split)
    columns = [
        "negative",
        "positive",
    ]
    test_ds = test_ds.map(
        lambda x: {
            "text": x["text"],
            LabelName.ltext: columns,
            LabelName.lint: [1 - x["label"], x["label"]],
        },
        remove_columns=test_ds.column_names,
    )
    return test_ds


def load_yahoo_answers_topics_dataset(split="test"):
    test_ds = datasets.load_dataset("mteb/yahoo_answers_topics", split=split)
    columns = test_ds.features["label"].names
    test_ds = test_ds.map(
        lambda x: {
            "text": x["text"],
            LabelName.ltext: columns,
            LabelName.lint: [x["label"] == i for i in range(len(columns))],
        },
        remove_columns=test_ds.column_names,
    )
    return test_ds


def load_news_dataset(split="test"):
    test_ds = datasets.load_dataset("mteb/news", split=split)
    columns = test_ds.features["label"].names
    test_ds = test_ds.map(
        lambda x: {
            "text": x["text"],
            LabelName.ltext: columns,
            LabelName.lint: [x["label"] == i for i in range(len(columns))],
        },
        remove_columns=test_ds.column_names,
    )
    return test_ds


def load_banking77_dataset(split="test"):
    test_ds = datasets.load_dataset("mteb/banking77", split=split)
    unique_labels = sorted(set(test_ds["label_text"]))
    columns = list(unique_labels)
    test_ds = test_ds.map(
        lambda x: {
            "text": x["text"],
            LabelName.ltext: columns,
            LabelName.lint: [x["label_text"] == i for i in columns],
        },
        remove_columns=test_ds.column_names,
    )
    return test_ds


def load_amazon_massive_intent_dataset(split="test"):
    test_ds = datasets.load_dataset(
        "mteb/MassiveIntentClassification", "en", split=split
    )
    unique_labels = sorted(set(test_ds["label"]))
    columns = list(unique_labels)
    test_ds = test_ds.map(
        lambda x: {
            "text": x["text"],
            LabelName.ltext: columns,
            LabelName.lint: [x["label"] == i for i in columns],
        },
        remove_columns=test_ds.column_names,
    )
    return test_ds


def load_amazon_massive_scenario_dataset(split="test"):
    test_ds = datasets.load_dataset("mteb/massive_scenario", "en", split=split)
    unique_labels = sorted(set(test_ds["label"]))
    columns = list(unique_labels)
    test_ds = test_ds.map(
        lambda x: {
            "text": x["text"],
            LabelName.ltext: columns,
            LabelName.lint: [x["label"] == i for i in columns],
        },
        remove_columns=test_ds.column_names,
    )
    return test_ds


def load_emotion_dataset(split="test"):
    test_ds = datasets.load_dataset("mteb/emotion", split=split)
    unique_labels = sorted(set(test_ds["label_text"]))
    columns = list(unique_labels)
    test_ds = test_ds.map(
        lambda x: {
            "text": x["text"],
            LabelName.ltext: columns,
            LabelName.lint: [x["label_text"] == i for i in columns],
        },
        remove_columns=test_ds.column_names,
    )
    return test_ds


def load_d_bpedia_dataset(split="test"):
    test_ds = datasets.load_dataset("mteb/d_bpedia", split=split)
    columns = test_ds.features["label"].names
    test_ds = test_ds.map(
        lambda x: {
            "text": x["text"],
            LabelName.ltext: columns,
            LabelName.lint: [x["label"] == i for i in range(len(columns))],
        },
        remove_columns=test_ds.column_names,
    )
    return test_ds


ds_mapping = {
    "toxic_conversations": load_toxic_conversations_dataset,
    "poem_sentiment": load_poem_sentiment_dataset,
    "movie_review_sentiment": load_movie_review_sentiment_dataset,
    "yahoo_answers_topics": load_yahoo_answers_topics_dataset,
    "news": load_news_dataset,
    # "banking77": load_banking77_dataset,
    "amazon_massive_intent": load_amazon_massive_intent_dataset,
    "emotion": load_emotion_dataset,
    "d_bpedia": load_d_bpedia_dataset,
    "amazon_massive_scenario": load_amazon_massive_scenario_dataset,
}
