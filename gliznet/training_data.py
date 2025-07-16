import random

import datasets

from . import LabelName

selected_columns = ["text", LabelName.ltext, LabelName.lint]


def load_mcq_dataset(
    ds_name: str,
    name: str,
    split: str = "train",
    text_column: str = "question",
    choices_column: str = "choices",
    answer_key_column: str = "answerKey",
):
    ds = datasets.load_dataset(ds_name, name, split=split)

    def mapper(x: dict[str, str]):
        return {
            "text": x[text_column],
            LabelName.ltext: x[choices_column]["text"],
            LabelName.lint: [
                i == x[answer_key_column] for i in x[choices_column]["label"]
            ],
        }

    ds = ds.map(mapper)
    return ds.select_columns(selected_columns)


def load_allenai_ai2_arc_easy():
    return load_mcq_dataset("allenai/ai2_arc", "ARC-Easy")


def load_allenai_ai2_arc_challenge():
    return load_mcq_dataset("allenai/ai2_arc", "ARC-Challenge")


def load_allenai_openbookqa():
    return load_mcq_dataset(
        "allenai/openbookqa",
        "additional",
        text_column="question_stem",
    )


def load_tau_commonsense_qa():
    return load_mcq_dataset(
        "tau/commonsense_qa",
        None,
    )


def load_Salesforce_cos_e():
    ds = datasets.load_dataset("Salesforce/cos_e", "v1.11", split="train")

    def mapper(x: dict[str, str]):
        return {
            "text": x["question"],
            LabelName.ltext: x["choices"],
            LabelName.lint: [i == x["answer"] for i in x["choices"]],
        }

    ds = ds.map(mapper)
    return ds.select_columns(selected_columns)


def load_onionmonster_dream():
    ds = datasets.load_dataset("onionmonster/dream", None, split="train")
    raws = [
        {
            "text": f"{query['question']}\n" + "\n".join(x["0"]),
            LabelName.ltext: query["choice"],
            LabelName.lint: [i == query["answer"] for i in query["choice"]],
        }
        for x in ds
        for query in x["1"]
    ]

    return datasets.Dataset.from_list(raws).select_columns(selected_columns)


def load_sagnikrayc_mctest():
    ds = datasets.load_dataset("sagnikrayc/mctest", "mc500", split="train")

    def mapper(x: dict[str, str]):
        return {
            "text": f"{x['question']}\n{x['story']}",
            LabelName.ltext: list(x["answer_options"].values()),
            LabelName.lint: [i == x["answer"] for i in x["answer_options"]],
        }

    ds = ds.map(mapper)
    return ds.select_columns(selected_columns)


def load_ehovy_race():
    ds = datasets.load_dataset("ehovy/race", "all", split="train")

    def mapper(x: dict[str, str]):
        return {
            "text": f"{x['question']}\n{x['article']}",
            LabelName.ltext: x["options"],
            LabelName.lint: [
                i == (ord(x["answer"]) - ord("A")) for i in range(len(x["options"]))
            ],
        }

    ds = ds.map(mapper)
    return ds.select_columns(selected_columns)


def load_sentence_transformers_wikihow():
    ds = datasets.load_dataset("sentence-transformers/wikihow", None, split="train")

    all_labels = list(set(ds["summary"]))

    def mapper(x: dict[str, str]):
        neg_count = random.randint(1, 4)
        neg_labels = random.sample(all_labels, neg_count)
        labels = [x["summary"]] + neg_labels
        random.shuffle(labels)
        return {
            "text": x["text"],
            LabelName.ltext: labels,
            LabelName.lint: [i == x["summary"] for i in labels],
        }

    ds = ds.map(mapper)
    return ds.select_columns(selected_columns)


def load_tasksource_cycic_classification():
    ds = datasets.load_dataset("tasksource/cycic_classification", None, split="train")

    def mapper(x: dict[str, str]):
        return {
            "text": x["question"],
            LabelName.ltext: ["false", "true"],
            LabelName.lint: [1 - int(x["correct_answer"]), int(x["correct_answer"])],
        }

    ds = ds.map(mapper)
    return ds.select_columns(selected_columns)


def load_ml4pubmed_pubmed_text_classification_cased():
    ds = datasets.load_dataset(
        "ml4pubmed/pubmed-text-classification-cased", None, split="train"
    )
    labels = list(set(ds["target"]))

    def mapper(x: dict[str, str]):
        return {
            "text": x["description_cln"],
            LabelName.ltext: labels,
            LabelName.lint: [x["target"] == label for label in labels],
        }

    ds = ds.map(mapper)
    return ds.select_columns(selected_columns)


def load_alexneakameni_qa_africa():
    ds = datasets.load_dataset("alexneakameni/qa_africa", None, split="train")

    def mapper(x: dict[str, str]):
        answer_choices = {k: v for k, v in x["answer_choices"].items() if v}
        return {
            "text": f"{x['question_text']}\n{x['explanation']}",
            LabelName.ltext: list(answer_choices.values()),
            LabelName.lint: [i in x["correct_answers"] for i in answer_choices],
        }

    ds = ds.map(mapper)
    return ds.select_columns(selected_columns)


additional_datasets = {
    "allenai_ai2_arc_easy": load_allenai_ai2_arc_easy,
    "allenai_ai2_arc_challenge": load_allenai_ai2_arc_challenge,
    "allenai_openbookqa": load_allenai_openbookqa,
    "tau_commonsense_qa": load_tau_commonsense_qa,
    "Salesforce_cos_e": load_Salesforce_cos_e,
    "onionmonster_dream": load_onionmonster_dream,
    "sagnikrayc_mctest": load_sagnikrayc_mctest,
    "ehovy_race": load_ehovy_race,
    "sentence_transformers_wikihow": load_sentence_transformers_wikihow,
    "tasksource_cycic_classification": load_tasksource_cycic_classification,
    "ml4pubmed_pubmed": load_ml4pubmed_pubmed_text_classification_cased,
    "alexneakameni_qa_africa": load_alexneakameni_qa_africa,
}
