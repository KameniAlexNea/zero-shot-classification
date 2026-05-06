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


def _load_gliclass(model_name: str, device: str = "cpu") -> GLiClassPipeline:
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
        classification_type="multi-label",
        device=device,
        progress_bar=False,
    )


gliclass_pipeline = _load_gliclass(GLICLASS_ID)


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
    gc_results = gliclass_pipeline(text, labels, threshold=0.0)[0]
    gc_scores = {r["label"]: round(r["score"], 4) for r in gc_results}
    gc_scores = _apply_threshold(gc_scores, threshold)

    return gz_scores, gc_scores


EXAMPLES = [
    # Few labels (2-3)
    ["I love pizza!", "food, travel", "multi-class", 0.0],
    [
        "The stock market crashed today after the Fed raised interest rates.",
        "finance, politics, sports",
        "multi-class",
        0.0,
    ],
    # Medium labels (5-7)
    [
        "One day I will see the world!",
        "travel, dreams, sport, science, politics",
        "multi-label",
        0.0,
    ],
    [
        "The government announced a new policy to reduce carbon emissions by 40% over the next decade.",
        "politics, environment, economics, technology, health",
        "multi-label",
        0.3,
    ],
    [
        "Scientists have discovered a new species of deep-sea fish that can produce its own light using bioluminescence.",
        "biology, technology, environment, space, medicine, chemistry",
        "multi-label",
        0.0,
    ],
    # Many labels (10+)
    [
        "After years of training and countless sacrifices, the athlete finally stood on the Olympic podium, tears streaming down her face as the national anthem played.",
        "sports, inspiration, perseverance, celebrity, politics, science, health, history, culture, entertainment, education",
        "multi-label",
        0.0,
    ],
    [
        "The newly released smartphone features a 200-megapixel camera, a foldable display, 24 hours of battery life, and an AI assistant that can draft emails, summarize documents, and manage your calendar.",
        "technology, business, artificial intelligence, photography, consumer electronics, finance, design, productivity, innovation, mobile, software, hardware",
        "multi-label",
        0.0,
    ],
]

with gr.Blocks(title="Zero-Shot Classification Comparison") as demo:
    gr.Markdown(
        "# Zero-Shot Classification: GliZNet vs GLiClass\n"
        "Compare **GliZNet** (`alexneakameni/gliznet-deberta-v3-base`) "
        "against **GLiClass** (`knowledgator/gliclass-base-v3.0`) side by side."
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

    btn = gr.Button("Classify", variant="primary")

    with gr.Row():
        gz_out = gr.Label(label="GliZNet (alexneakameni)")
        gc_out = gr.Label(label="GLiClass (knowledgator)")

    btn.click(
        fn=classify,
        inputs=[text_input, labels_input, cls_type, threshold],
        outputs=[gz_out, gc_out],
    )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[text_input, labels_input, cls_type, threshold],
        outputs=[gz_out, gc_out],
        fn=classify,
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch()
