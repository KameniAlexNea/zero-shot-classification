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
All models evaluated under identical ranking metrics.

### GliZNet vs GLiClass (direct competitor)

GLiClass is the closest published competitor — it also encodes text and labels jointly in a single forward pass.

| Model | MRR | Hit@1 | Hit@3 | Hit@5 | NDCG@10 |
|---|---|---|---|---|---|
| `knowledgator/gliclass-base-v3.0` | 0.927 | 0.862 | 0.996 | 0.999 | 0.920 |
| **GliZNet-deberta-v3-base (ours)** | **0.966** | **0.935** | 0.985 | **1.000** | **0.942** |

**Δ GliZNet − GLiClass**: MRR +0.039 · Hit@1 +0.073 · NDCG@10 +0.023

### GliZNet vs sentence-embedding baselines

Independent text/label embedding with cosine similarity

| Model | MRR | Hit@1 | Hit@3 | Hit@5 | NDCG@10 | ROC-AUC | Avg Precision |
|---|---|---|---|---|---|---|---|
| `OrdalieTech/Solon-embeddings-large-0.1` | 0.914 | 0.843 | 0.989 | 1.000 | 0.895 | 0.708 | 0.800 |
| `jinaai/jina-embeddings-v5-text-small` | 0.936 | 0.883 | 0.995 | 1.000 | 0.914 | 0.759 | 0.832 |
| `microsoft/harrier-oss-v1-0.6b` | 0.915 | 0.840 | 0.992 | 1.000 | 0.897 | 0.716 | 0.804 |
| `intfloat/e5-large-v2` | 0.931 | 0.871 | 0.994 | 1.000 | 0.911 | 0.753 | 0.828 |
| **GliZNet-deberta-v3-base (ours)** | **0.966** | **0.935** | 0.985 | **1.000** | **0.942** | **0.825** | **0.874** |

*All GliZNet results from `checkpoint-850` (best checkpoint by eval loss).*

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
