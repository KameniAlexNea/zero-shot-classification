import gradio as gr
from gliznet.predictor import ZeroShotClassificationPipeline
from gliclass import GLiClassModel
from gliclass import ZeroShotClassificationPipeline as GLiClassPipeline
from gliclass.model import GLiClassModelConfig
from transformers import AutoTokenizer
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

GLIZNET_ID = "alexneakameni/gliznet-deberta-v3-base"
GLICLASS_ID = "knowledgator/gliclass-base-v3.0"

gliznet_pipeline = ZeroShotClassificationPipeline.from_pretrained(
    GLIZNET_ID, classification_type="multi-label", device="cpu"
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
        return {}, {}

    # GliZNet
    gz_output = gliznet_pipeline(
        text, labels, threshold=None, classification_type=classification_type
    )
    gz_scores = {item.label: round(item.score, 4) for item in gz_output.labels}
    gz_scores = _apply_threshold(gz_scores, threshold)

    # GLiClass (threshold=0.0 returns all labels; we filter manually below)
    gc_pipeline = gliclass_pipelines.get(classification_type, gliclass_pipelines["multi-label"])
    gc_results = gc_pipeline(text, labels, threshold=0.0,)[0]
    gc_scores = {r["label"]: round(r["score"], 4) for r in gc_results}
    gc_scores = _apply_threshold(gc_scores, threshold)

    return gz_scores, gc_scores


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

with gr.Blocks(title="Zero-Shot Classification Comparison") as demo:
    gr.Markdown(
        "# Zero-Shot Classification: GliZNet vs GLiClass\n"
        "Compare **GliZNet** (`alexneakameni/gliznet-deberta-v3-base`) "
        "against **GLiClass** (`knowledgator/gliclass-base-v3.0`) side by side.\n\n"
        "Click an example below to see cached predictions instantly."
    )

    with gr.Row():
        text_input = gr.Textbox(
            label="Text", lines=4, placeholder="Enter text to classify..."
        )
        labels_input = gr.Textbox(
            label="Labels (comma-separated)",
            placeholder="positive, negative, neutral",
        )

    with gr.Row():
        cls_type = gr.Radio(
            ["multi-label", "multi-class"],
            label="Classification Type",
            value="multi-label",
        )
        threshold = gr.Slider(
            0.0, 1.0, value=0.0, step=0.05, label="Threshold (0 = show all)"
        )

    with gr.Row():
        expected_box = gr.Textbox(label="Expected Labels", interactive=False)
        why_not_box = gr.Textbox(label="Why Not?", interactive=False, lines=3)

    btn = gr.Button("Classify", variant="primary")

    with gr.Row():
        gz_out = gr.Label(label="GliZNet (alexneakameni)")
        gc_out = gr.Label(label="GLiClass (knowledgator)")

    btn.click(
        fn=classify,
        inputs=[text_input, labels_input, cls_type, threshold],
        outputs=[gz_out, gc_out],
    )

    def classify_example(text, labels_str, classification_type, threshold, _expected, _why_not):
        return classify(text, labels_str, classification_type, threshold)

    gr.Examples(
        examples=EXAMPLES,
        inputs=[text_input, labels_input, cls_type, threshold, expected_box, why_not_box],
        outputs=[gz_out, gc_out],
        fn=classify_example,
        cache_examples=True,
    )

if __name__ == "__main__":
    demo.launch()
