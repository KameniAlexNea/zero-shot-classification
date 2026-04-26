---
language:
  - en
license: apache-2.0
tags:
  - zero-shot-classification
  - text-classification
  - deberta
  - gliznet
  - joint-encoding
  - contrastive-learning
datasets:
  - alexneakameni/ZSHOT-HARDSET-v2
metrics:
  - mrr
  - hit_at_k
  - ndcg
base_model: microsoft/deberta-v3-base
pipeline_tag: zero-shot-classification
---

# GliZNet — DeBERTa-v3-base

**GliZNet** (Generalized Zero-Shot Network) is a zero-shot text classification model that processes the input text and **all candidate labels jointly in a single forward pass**, achieving O(1) inference complexity regardless of the number of labels.

Built on top of [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base), GliZNet encodes text and labels together in one sequence, generates label-aware contextual embeddings, and scores each label via cosine similarity. A hybrid loss combining multi-label softmax cross-entropy (primary), auxiliary BCE with decoupled temperature, and optional label repulsion sharpens discrimination between semantically similar labels.

> **Paper**: *GliZNet: A Novel Architecture for Zero-Shot Text Classification*  
> Alex Kameni (Ivalua / Massy, France)
>
> **Code**: [github.com/KameniAlexNea/zero-shot-classification](https://github.com/KameniAlexNea/zero-shot-classification)  
> **Synthetic data generation**: [github.com/KameniAlexNea/generate-gliznet-data](https://github.com/KameniAlexNea/generate-gliznet-data)

---

## Model Details

| Property | Value |
|---|---|
| Backbone | `microsoft/deberta-v3-base` (~184 M params) |
| Projection dim | 1024 |
| Similarity metric | Cosine |
| Max sequence length | 1024 tokens |
| Label separator token | `[LAB]` |
| Label representation | Average of label token hidden states |
| Training precision | `bfloat16` |
| Model type ID | `gliznet` |

### Architecture Summary

```
Input: [CLS] <text tokens> [SEP] <label_1 tokens> [LAB] <label_2 tokens> [LAB] ... [PAD]
         │
    DeBERTa-v3-base  (frozen/fine-tuned)
         │
    ┌────┴────────────────────────┐
    │  Text repr (CLS hidden)     │  Label reprs (avg per label span)
    └────────────────┬────────────┘
              Linear projection → dim 1024
                     │
              Cosine similarity
                     │
              Learnable temperature scale
                     │
              MultiLabel-Softmax + BCE + Repulsion (training)
```

---

## Usage

### With `ZeroShotClassificationPipeline`

```python
import torch
from gliznet.model import GliZNetForSequenceClassification
from gliznet.tokenizer import GliZNETTokenizer
from gliznet.predictor import ZeroShotClassificationPipeline

model = GliZNetForSequenceClassification.from_pretrained("alexneakameni/gliznet-deberta-v3-base")
model = model.to(torch.bfloat16)
tokenizer = GliZNETTokenizer.from_pretrained("alexneakameni/gliznet-deberta-v3-base")

pipeline = ZeroShotClassificationPipeline(
    model, tokenizer,
    classification_type="multi-label",   # or "multi-class"
    device="cuda",
)

text = "Scientists discover a new exoplanet orbiting a distant star."
labels = ["astronomy", "politics", "cooking", "space exploration", "finance"]

result = pipeline(text, labels)
for ls in sorted(result.labels, key=lambda x: -x.score):
    print(f"  {ls.label:<25} {ls.score:.3f}")
```

### With `from_pretrained` + HuggingFace AutoModel

```python
from transformers import AutoModel, AutoConfig
import gliznet  # triggers Auto* registration

config = AutoConfig.from_pretrained("alexneakameni/gliznet-deberta-v3-base")
model  = AutoModel.from_pretrained("alexneakameni/gliznet-deberta-v3-base")
```

---

## Performance

Evaluated on the held-out test split of **ZSHOT-HARDSET-v2** (1,322 samples, up to 20 labels per sample).

| Metric | Score |
|---|---|
| MRR | **0.966** |
| Hit@1 | **0.935** |
| Hit@3 | **0.985** |
| Hit@5 | **1.000** |
| NDCG@10 | **0.942** |

*Metrics computed on `checkpoint-850` (best checkpoint by eval loss from epoch 1 of 10).*

---

## Training Details

| Setting | Value |
|---|---|
| Dataset | `alexneakameni/ZSHOT-HARDSET-v2` |
| Train / Val / Test split | 54,289 / 1,188 / 1,322 |
| Optimizer | AdamW |
| Learning rate | 1e-4 (cosine schedule, 5% warmup) |
| Weight decay | 1e-3 |
| Batch size | 16 × 2 GPUs × 4 grad. accum. = **128 effective** |
| Epochs | 10 (early stopping, patience=3) |
| Precision | bf16 |
| Distributed training | DeepSpeed ZeRO-2 via `accelerate launch` |
| Hardware | 2 × NVIDIA GPU |
| Loss | Multi-label softmax (weight 1.0) + auxiliary BCE (weight 1.0) + label repulsion (weight 0.1, disabled by default) |
| Max labels per sample | 20 |

---

## Tokenizer

`GliZNETTokenizer` wraps the DeBERTa-v3 sentencepiece tokenizer and adds a custom `[LAB]` separator token. The input format is:

```
[CLS] <text> [SEP] <label_1> [LAB] <label_2> [LAB] ... <label_n> [LAB] [PAD]*
```

The `lmask` tensor (label mask) assigns 0 to text tokens and unique integers 1…n to each label's tokens, allowing the model to pool each label independently.

---

## Limitations

- Trained on synthetic English data; performance on specialized domains (legal, medical) or non-English text may degrade.
- Best results when all candidate labels fit within 1024 tokens. Very large label sets should be batched.
- The model does not produce calibrated probabilities; scores are cosine similarities scaled by a learned temperature.

---

## Citation

```bibtex
@article{kameni2025gliznet,
  title   = {GliZNet: A Novel Architecture for Zero-Shot Text Classification},
  author  = {Alex Kameni},
  year    = {2025},
  note    = {Preprint. Code: https://github.com/KameniAlexNea/zero-shot-classification}
}
```

---

## License

Apache 2.0
