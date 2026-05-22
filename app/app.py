import json
import random
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    ndcg_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from gliznet.predictor import ZeroShotClassificationPipeline
from gliclass import GLiClassModel
from gliclass import ZeroShotClassificationPipeline as GLiClassPipeline
from gliclass.model import GLiClassModelConfig
from transformers import AutoTokenizer
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

GLIZNET_ID = "alexneakameni/gliznet-deberta-v3-base"
GLIZNET_MODERN_ID = "alexneakameni/gliznet-ModernBERT-base"
GLICLASS_ID = "knowledgator/gliclass-base-v3.0"
EVAL_JSON = Path(__file__).parent / "eval_examples.json"

gliznet_pipeline = ZeroShotClassificationPipeline.from_pretrained(
    GLIZNET_ID, classification_type="multi-label", device="cpu"
)
gliznet_modern_pipeline = ZeroShotClassificationPipeline.from_pretrained(
    GLIZNET_MODERN_ID, classification_type="multi-label", device="cpu"
)


def _load_gliclass(model_name: str, classification_type: str = "multi-label", device: str = "cpu") -> GLiClassPipeline:
    """Load GLiClass with manual weight loading.
    transformers 5.x from_pretrained silently fails to apply checkpoint weights
    for unregistered model types, so we load them manually via load_state_dict."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = GLiClassModelConfig.from_pretrained(model_name)
    config.pad_token_id = tokenizer.pad_token_id

    _orig_tie_weights = GLiClassModel.tie_weights
    GLiClassModel.tie_weights = lambda self, **kwargs: _orig_tie_weights(self)

    model = GLiClassModel(config)
    ckpt_path = hf_hub_download(model_name, "model.safetensors")
    model.load_state_dict(load_file(ckpt_path), strict=True)

    return GLiClassPipeline(
        model, tokenizer,
        classification_type=classification_type,
        device=device,
        progress_bar=False,
    )


gliclass_pipelines = {
    "multi-label": _load_gliclass(GLICLASS_ID, "multi-label"),
    "multi-class": _load_gliclass(GLICLASS_ID, "single-label"),
}


def _apply_threshold(scores: dict, threshold: float) -> dict:
    if threshold > 0.0:
        filtered = {k: v for k, v in scores.items() if v >= threshold}
        return filtered if filtered else scores
    return scores


def classify(text: str, labels_str: str, classification_type: str, threshold: float):
    labels = [l.strip() for l in labels_str.split(",") if l.strip()]
    if not text or not labels:
        return {}, {}, {}

    # GliZNet DeBERTa
    gz_output = gliznet_pipeline(
        text, labels, threshold=None, classification_type=classification_type
    )
    gz_scores = {item.label: round(item.score, 4) for item in gz_output.labels}
    gz_scores = _apply_threshold(gz_scores, threshold)

    # GliZNet ModernBERT
    gzm_output = gliznet_modern_pipeline(
        text, labels, threshold=None, classification_type=classification_type
    )
    gzm_scores = {item.label: round(item.score, 4) for item in gzm_output.labels}
    gzm_scores = _apply_threshold(gzm_scores, threshold)

    # GLiClass (threshold=0.0 returns all labels; we filter manually below)
    gc_pipeline = gliclass_pipelines.get(classification_type, gliclass_pipelines["multi-label"])
    gc_results = gc_pipeline(text, labels, threshold=0.0,)[0]
    gc_scores = {r["label"]: round(r["score"], 4) for r in gc_results}
    gc_scores = _apply_threshold(gc_scores, threshold)

    return gz_scores, gzm_scores, gc_scores


def _load_eval_examples():
    if EVAL_JSON.exists():
        with open(EVAL_JSON) as f:
            return json.load(f)
    return []


EVAL_EXAMPLES = _load_eval_examples()


EXAMPLES = [
    # [text, labels, type, threshold, expected, why_not]

    # ── Fine-grained sentiment (GliZNet's strength) ──────────────────────────
    [
        "The restaurant was okay — nothing special, but the pasta was edible and the waiter tried his best.",
        "very positive, positive, neutral, negative, very negative",
        "multi-class",
        0.0,
        "neutral",
        "'positive' — the praise is faint and hedged ('okay', 'tried his best'), not genuine enthusiasm. "
        "'negative' — no complaint is made; the tone is resigned acceptance, not dissatisfaction.",
    ],
    [
        "I was hoping for more, honestly. The build quality is fine but the battery barely lasts half a day.",
        "very positive, positive, neutral, negative, very negative",
        "multi-class",
        0.0,
        "negative",
        "'neutral' — 'hoping for more' and 'barely lasts' express clear disappointment, not indifference. "
        "'very negative' — the reviewer concedes 'build quality is fine', softening the overall stance.",
    ],

    # ── Semantically close labels ────────────────────────────────────────────
    [
        "The CEO announced record quarterly profits while simultaneously laying off 2,000 employees to cut costs.",
        "corporate restructuring, financial success, employee welfare, economic growth, labor dispute",
        "multi-label",
        0.3,
        "corporate restructuring, financial success",
        "'employee welfare' — layoffs are the opposite of welfare; the text describes harm, not care. "
        "'economic growth' — profits are company-specific, not macroeconomic growth. "
        "'labor dispute' — no conflict or negotiation is described; the layoffs are unilateral.",
    ],
    [
        "New research shows that moderate coffee consumption may reduce the risk of Alzheimer's disease by up to 30%.",
        "medical research, nutrition advice, drug development, disease prevention, public health policy",
        "multi-label",
        0.3,
        "medical research, disease prevention",
        "'nutrition advice' — the text reports a study finding, not a dietary recommendation. "
        "'drug development' — coffee is not a drug being developed; this is observational research. "
        "'public health policy' — no policy or regulation is discussed.",
    ],

    # ── Rhetorical stance / intent ───────────────────────────────────────────
    [
        "Sure, let's just keep dumping plastic into the ocean. That'll definitely fix everything.",
        "environmental activism, sincere optimism, sarcasm, policy proposal, scientific analysis",
        "multi-class",
        0.0,
        "sarcasm",
        "'environmental activism' — while the topic is environmental, the stance is ironic commentary, not a call to action. "
        "'sincere optimism' — 'That'll definitely fix everything' is clearly ironic. "
        "'policy proposal' — no concrete policy is proposed.",
    ],
    [
        "While the opposition raises valid concerns about cost, the long-term savings from renewable energy "
        "infrastructure far outweigh the initial investment, as demonstrated by Denmark's 40-year track record.",
        "political argument, scientific evidence, emotional appeal, balanced reporting, policy advocacy",
        "multi-label",
        0.3,
        "political argument, policy advocacy",
        "'balanced reporting' — the author takes a clear side ('far outweigh'), this is not neutral reporting. "
        "'scientific evidence' — Denmark's track record is a policy outcome, not a scientific experiment. "
        "'emotional appeal' — the argument relies on data and logic, not emotion.",
    ],

    # ── Many labels with hard negatives ──────────────────────────────────────
    [
        "After years of training and countless sacrifices, the athlete finally stood on the Olympic podium, "
        "tears streaming down her face as the national anthem played.",
        "athletic achievement, personal sacrifice, patriotism, emotional moment, celebrity gossip, "
        "sports injury, political protest, entertainment review, historical analysis, travel experience",
        "multi-label",
        0.3,
        "athletic achievement, personal sacrifice, patriotism, emotional moment",
        "'celebrity gossip' — the text is a narrative of achievement, not tabloid speculation. "
        "'sports injury' — sacrifice here is metaphorical (time, effort), not physical injury. "
        "'political protest' — the anthem scene is patriotic pride, not a protest.",
    ],
]


# ── Helpers for evaluation ───────────────────────────────────────────────────

def _get_ranked_scores(text, labels):
    """Return ranked (label, score) lists for all models, sorted by score desc.
    Always uses multi-label mode so every label gets a score."""
    gz_output = gliznet_pipeline(text, labels, threshold=None, classification_type="multi-label")
    gz_ranked = sorted(
        [(item.label, item.score) for item in gz_output.labels],
        key=lambda x: x[1], reverse=True,
    )

    gzm_output = gliznet_modern_pipeline(text, labels, threshold=None, classification_type="multi-label")
    gzm_ranked = sorted(
        [(item.label, item.score) for item in gzm_output.labels],
        key=lambda x: x[1], reverse=True,
    )

    gc_pipeline = gliclass_pipelines["multi-label"]
    gc_results = gc_pipeline(text, labels, threshold=0.0)[0]
    gc_ranked = sorted(
        [(r["label"], r["score"]) for r in gc_results],
        key=lambda x: x[1], reverse=True,
    )

    return gz_ranked, gzm_ranked, gc_ranked


def _example_metrics(ranked, expected_set, threshold):
    """Compute all metrics for one example using sklearn."""
    labels = [l for l, _ in ranked]
    scores = [s for _, s in ranked]
    y_true = np.array([1 if l in expected_set else 0 for l in labels])
    y_pred = np.array([1 if s >= threshold else 0 for s in scores])
    y_scores = np.array(scores)

    # MRR — reciprocal rank of first relevant label
    mrr = 0.0
    for i, lab in enumerate(labels):
        if lab in expected_set:
            mrr = 1.0 / (i + 1)
            break

    # Hit@k
    hit1 = float(labels[0] in expected_set) if labels else 0.0
    hit3 = float(any(l in expected_set for l in labels[:3]))

    # NDCG via sklearn (needs 2D arrays)
    y_true_2d = y_true.reshape(1, -1)
    y_scores_2d = y_scores.reshape(1, -1)
    ndcg3 = ndcg_score(y_true_2d, y_scores_2d, k=3)
    ndcg5 = ndcg_score(y_true_2d, y_scores_2d, k=5)
    ndcg_full = ndcg_score(y_true_2d, y_scores_2d)

    # Precision / Recall / F1 at threshold
    precision = precision_score(y_true, y_pred, zero_division=0.0)
    recall = recall_score(y_true, y_pred, zero_division=0.0)
    f1 = f1_score(y_true, y_pred, zero_division=0.0)

    return {
        "hit@1": hit1, "hit@3": hit3, "mrr": mrr,
        "ndcg@3": ndcg3, "ndcg@5": ndcg5, "ndcg": ndcg_full,
        "precision": precision, "recall": recall, "f1": f1,
        "y_true": y_true, "y_scores": y_scores,
    }


def _sample_random():
    """Pick a random eval example and return fields + multi-label predictions."""
    if not EVAL_EXAMPLES:
        return "", "", "multi-label", "", "", {}, {}, {}
    ex = random.choice(EVAL_EXAMPLES)
    labels_str = ", ".join(ex["labels"])
    expected_str = ", ".join(ex["expected"])
    why_not_lines = [f"'{k}' — {v}" for k, v in ex.get("not_labels_explained", {}).items()]
    why_not_str = "\n".join(why_not_lines)

    gz_ranked, gzm_ranked, gc_ranked = _get_ranked_scores(ex["text"], ex["labels"])
    gz_display = {l: round(s, 4) for l, s in gz_ranked}
    gzm_display = {l: round(s, 4) for l, s in gzm_ranked}
    gc_display = {l: round(s, 4) for l, s in gc_ranked}

    return (
        ex["text"], labels_str, "multi-label",
        expected_str, why_not_str,
        gz_display, gzm_display, gc_display,
    )


def _run_full_eval(threshold, progress=gr.Progress()):
    """Evaluate all models on all examples with ranking + classification metrics."""
    if not EVAL_EXAMPLES:
        return "No examples loaded.", pd.DataFrame()

    rows = []
    metric_keys = ("hit@1", "hit@3", "mrr", "ndcg@3", "ndcg@5", "ndcg", "precision", "recall", "f1")
    gz_agg = {k: [] for k in metric_keys}
    gzm_agg = {k: [] for k in metric_keys}
    gc_agg = {k: [] for k in metric_keys}
    gz_all_y, gz_all_scores = [], []
    gzm_all_y, gzm_all_scores = [], []
    gc_all_y, gc_all_scores = [], []

    for ex in progress.tqdm(EVAL_EXAMPLES, desc="Evaluating"):
        expected = set(ex["expected"])
        gz_ranked, gzm_ranked, gc_ranked = _get_ranked_scores(ex["text"], ex["labels"])

        gz_m = _example_metrics(gz_ranked, expected, threshold)
        gzm_m = _example_metrics(gzm_ranked, expected, threshold)
        gc_m = _example_metrics(gc_ranked, expected, threshold)

        for k in metric_keys:
            gz_agg[k].append(gz_m[k])
            gzm_agg[k].append(gzm_m[k])
            gc_agg[k].append(gc_m[k])

        # Collect for global ROC AUC
        gz_all_y.extend(gz_m["y_true"])
        gz_all_scores.extend(gz_m["y_scores"])
        gzm_all_y.extend(gzm_m["y_true"])
        gzm_all_scores.extend(gzm_m["y_scores"])
        gc_all_y.extend(gc_m["y_true"])
        gc_all_scores.extend(gc_m["y_scores"])

        # Per-example row: show top-k until all expected labels are covered
        def _top_until_covered(ranked, expected):
            labels = [l for l, _ in ranked]
            found = set()
            k = 0
            for i, l in enumerate(labels):
                k = i + 1
                if l in expected:
                    found.add(l)
                if found == expected:
                    break
            return ", ".join(l for l, _ in ranked[:k])

        gz_top = _top_until_covered(gz_ranked, expected)
        gzm_top = _top_until_covered(gzm_ranked, expected)
        gc_top = _top_until_covered(gc_ranked, expected)

        rows.append({
            "text": ex["text"][:80] + ("…" if len(ex["text"]) > 80 else ""),
            "expected": ", ".join(sorted(expected)),
            "GliZNet-DeBERTa ranking": gz_top,
            "GliZNet-ModernBERT ranking": gzm_top,
            "GLiClass ranking": gc_top,
        })

    n = len(EVAL_EXAMPLES)

    try:
        gz_auc = roc_auc_score(gz_all_y, gz_all_scores)
    except ValueError:
        gz_auc = float("nan")
    try:
        gzm_auc = roc_auc_score(gzm_all_y, gzm_all_scores)
    except ValueError:
        gzm_auc = float("nan")
    try:
        gc_auc = roc_auc_score(gc_all_y, gc_all_scores)
    except ValueError:
        gc_auc = float("nan")

    ranking_keys = [
        ("Hit@1", "hit@1"), ("Hit@3", "hit@3"), ("MRR", "mrr"),
        ("NDCG@3", "ndcg@3"), ("NDCG@5", "ndcg@5"), ("NDCG", "ndcg"),
    ]
    clf_keys = [
        ("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"),
    ]

    lines = [
        f"## Results — {n} examples, threshold={threshold}\n",
        "### Ranking Metrics\n",
        "| Metric | GliZNet-DeBERTa | GliZNet-ModernBERT | GLiClass |",
        "|--------|-----------------|--------------------|-----------|\n",
    ]
    for display, key in ranking_keys:
        gz_val = np.mean(gz_agg[key])
        gzm_val = np.mean(gzm_agg[key])
        gc_val = np.mean(gc_agg[key])
        lines.append(f"| {display} | {gz_val:.4f} | {gzm_val:.4f} | {gc_val:.4f} |")

    lines += [
        "",
        f"### Classification Metrics (threshold={threshold})\n",
        "| Metric | GliZNet-DeBERTa | GliZNet-ModernBERT | GLiClass |",
        "|--------|-----------------|--------------------|-----------|\n",
    ]
    for display, key in clf_keys:
        gz_val = np.mean(gz_agg[key])
        gzm_val = np.mean(gzm_agg[key])
        gc_val = np.mean(gc_agg[key])
        lines.append(f"| {display} | {gz_val:.4f} | {gzm_val:.4f} | {gc_val:.4f} |")

    lines += [
        "",
        "### ROC AUC (micro, across all label decisions)\n",
        "| Metric | GliZNet-DeBERTa | GliZNet-ModernBERT | GLiClass |",
        "|--------|-----------------|--------------------|-----------|\n",
        f"| ROC AUC | {gz_auc:.4f} | {gzm_auc:.4f} | {gc_auc:.4f} |",
    ]

    return "\n".join(lines), pd.DataFrame(rows)


# ── UI ───────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Zero-Shot Classification: GliZNet vs GLiClass") as demo:
    gr.Markdown(
        "# Zero-Shot Classification: GliZNet vs GLiClass\n"
        "Compare **GliZNet-DeBERTa** (`alexneakameni/gliznet-deberta-v3-base`), "
        "**GliZNet-ModernBERT** (`alexneakameni/gliznet-ModernBERT-base`), "
        "and **GLiClass** (`knowledgator/gliclass-base-v3.0`)."
    )

    with gr.Tabs():
        # ── Tab 1: Interactive playground ─────────────────────────────────
        with gr.TabItem("Playground"):
            gr.Markdown("Enter your own text and labels, or click **Random Example** to load one from the eval set.")

            with gr.Row():
                text_input = gr.Textbox(label="Text", lines=4, placeholder="Enter text to classify...")
                labels_input = gr.Textbox(label="Labels (comma-separated)", placeholder="positive, negative, neutral")

            with gr.Row():
                cls_type = gr.Radio(["multi-label", "multi-class"], label="Classification Type", value="multi-label")
                threshold = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Threshold (0 = show all)")

            with gr.Row():
                expected_box = gr.Textbox(label="Expected Labels", interactive=False)
                why_not_box = gr.Textbox(label="Why Not? (hard negatives)", interactive=False, lines=3)

            with gr.Row():
                random_btn = gr.Button("🎲 Random Example", variant="secondary")
                classify_btn = gr.Button("Classify", variant="primary")

            with gr.Row():
                gz_out = gr.Label(label="GliZNet-DeBERTa")
                gzm_out = gr.Label(label="GliZNet-ModernBERT")
                gc_out = gr.Label(label="GLiClass")

            classify_btn.click(
                fn=classify,
                inputs=[text_input, labels_input, cls_type, threshold],
                outputs=[gz_out, gzm_out, gc_out],
            )

            random_btn.click(
                fn=_sample_random,
                inputs=[],
                outputs=[text_input, labels_input, cls_type, expected_box, why_not_box, gz_out, gzm_out, gc_out],
            )

            def _classify_example(text, labels_str, classification_type, threshold, _expected, _why_not):
                return classify(text, labels_str, classification_type, threshold)

            gr.Examples(
                examples=EXAMPLES,
                inputs=[text_input, labels_input, cls_type, threshold, expected_box, why_not_box],
                outputs=[gz_out, gzm_out, gc_out],
                fn=_classify_example,
                cache_examples=True,
            )

        # ── Tab 2: Batch evaluation ──────────────────────────────────────
        with gr.TabItem("Evaluation"):
            gr.Markdown(
                f"Run all models on all **{len(EVAL_EXAMPLES)}** examples and compute "
                "ranking metrics (Hit@k, MRR, NDCG) and classification metrics "
                "(Precision, Recall, F1, ROC AUC).\n\n"
                "All examples are evaluated as **multi-label**: every candidate label "
                "gets a score, then we measure how well the expected labels are ranked "
                "and how accurately they are selected at the chosen threshold."
            )

            eval_threshold = gr.Slider(
                0.1, 0.9, value=0.5, step=0.05,
                label="Multi-label threshold (labels with score ≥ threshold are predicted positive)",
            )

            eval_btn = gr.Button("▶ Run Evaluation", variant="primary")

            eval_metrics = gr.Markdown("*Click 'Run Evaluation' to start.*")
            eval_table = gr.Dataframe(
                label="Per-example results (ranked labels until all expected are covered)",
                interactive=True,
                wrap=True,
            )

            eval_btn.click(
                fn=_run_full_eval,
                inputs=[eval_threshold],
                outputs=[eval_metrics, eval_table],
            )

if __name__ == "__main__":
    demo.launch()
