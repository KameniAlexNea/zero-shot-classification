import datasets
from typing import List, Optional, Callable, Any


class LabelName:
    ltext = "ltext"
    lint = "lint"


def _load_dataset_generic(
    dataset_path: str,
    subset: Optional[str] = None,
    split: str = "test",
    label_column: str = "label",
    label_mapper: Optional[Callable[[Any, List[str]], List[bool]]] = None,
    custom_columns: Optional[List[str]] = None,
) -> datasets.Dataset:
    """
    Generic dataset loader that handles common patterns for MTEB datasets.
    
    Args:
        dataset_path: HuggingFace dataset path
        subset: Dataset subset (e.g., "en" for amazon_massive_intent)
        split: Dataset split to load
        label_column: Column name containing labels
        label_mapper: Function to map labels to binary arrays
        custom_columns: Predefined list of columns (overrides automatic detection)
    """
    # Load dataset
    if subset:
        test_ds = datasets.load_dataset(dataset_path, subset, split=split)
    else:
        test_ds = datasets.load_dataset(dataset_path, split=split)
    
    # Determine columns/labels
    if custom_columns:
        columns = custom_columns
    elif hasattr(test_ds.features[label_column], 'names'):
        # For datasets with named labels (e.g., poem_sentiment, yahoo_answers)
        columns = test_ds.features[label_column].names
    elif label_column == "label_text":
        # For datasets using label_text column (banking77, emotion)
        unique_labels = set(test_ds[label_column])
        columns = list(unique_labels)
    elif label_column == "label":
        # For datasets using label column with string values (amazon_massive_intent)
        unique_labels = set(test_ds[label_column])
        columns = list(unique_labels)
    else:
        raise ValueError(f"Unable to determine columns for label_column: {label_column}")
    
    # Default label mapper
    if label_mapper is None:
        if label_column == "label_text":
            def label_mapper(x, cols):
                return [x[label_column] == col for col in cols]
        elif label_column == "label" and isinstance(test_ds[0][label_column], str):
            def label_mapper(x, cols):
                return [x[label_column] == col for col in cols]
        elif label_column == "label" and isinstance(test_ds[0][label_column], int):
            def label_mapper(x, cols):
                return [x[label_column] == i for i in range(len(cols))]
        else:
            # For toxic_conversations - direct label value
            def label_mapper(x, cols):
                return [x[label_column]]
    
    # Transform dataset
    test_ds = test_ds.map(
        lambda x: {
            "text": x["text"],
            LabelName.ltext: columns,
            LabelName.lint: label_mapper(x, columns),
        },
        remove_columns=test_ds.column_names,
    )
    
    return test_ds


def load_toxic_conversations_dataset():
    def toxic_label_mapper(x, cols):
        return [x["label"]]
    
    return _load_dataset_generic(
        "mteb/toxic_conversations",
        custom_columns=["toxic"],
        label_mapper=toxic_label_mapper
    )


def load_poem_sentiment_dataset():
    return _load_dataset_generic("mteb/poem_sentiment")


def load_movie_review_sentiment_dataset():
    return _load_dataset_generic(
        "mteb/movie_review_sentiment",
        custom_columns=["positive"]
    )


def load_yahoo_answers_topics_dataset():
    return _load_dataset_generic("mteb/yahoo_answers_topics")


def load_news_dataset():
    return _load_dataset_generic("mteb/news")


def load_banking77_dataset():
    return _load_dataset_generic("mteb/banking77", label_column="label_text")


def load_amazon_massive_intent_dataset():
    return _load_dataset_generic("mteb/amazon_massive_intent", subset="en")


def load_emotion_dataset():
    return _load_dataset_generic("mteb/emotion", label_column="label_text")


def load_d_bpedia_dataset():
    return _load_dataset_generic("mteb/d_bpedia")


ds_mapping = {
    "toxic_conversations": load_toxic_conversations_dataset,
    "poem_sentiment": load_poem_sentiment_dataset,
    "movie_review_sentiment": load_movie_review_sentiment_dataset,
    "yahoo_answers_topics": load_yahoo_answers_topics_dataset,
    "news": load_news_dataset,
    "banking77": load_banking77_dataset,
    "amazon_massive_intent": load_amazon_massive_intent_dataset,
    "emotion": load_emotion_dataset,
    "d_bpedia": load_d_bpedia_dataset,
}
