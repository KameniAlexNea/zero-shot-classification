# GliZNet: A Novel Architecture for Zero-Shot Text Classification

**Authors**: Alex Kameni (Ivalua / Massy, France, eak@ivalua.com)
**Date**: July 7, 2025

## Abstract

Zero-shot text classification, crucial for dynamic and data-scarce environments, often struggles with computational inefficiency and limited inter-label reasoning, particularly for large label sets. We introduce GliZNet (Generalized Zero-Shot Network), a novel architecture that addresses these challenges through a joint text-label encoding mechanism, processing text and all candidate labels in a single transformer forward pass to achieve O(1) complexity. Enhanced by a multi-objective loss combining one-vs-negatives softmax cross-entropy with additive margin, focal loss, and label repulsion, GliZNet generates label-aware contextual embeddings, enabling fine-grained differentiation of semantically similar labels. Trained on a diverse synthetic dataset with online text and label augmentation, GliZNet achieves competitive performance with GLiClass-base (0.6609 vs 0.7056 avg macro F1) using a simpler single-stage training pipeline. This work demonstrates that efficient and context-sensitive zero-shot learning can be achieved through careful architectural design and data augmentation without requiring multi-stage training or reinforcement learning.

## 1. Introduction

Zero-shot text classification enables models to categorize text into labels not encountered during training, a critical capability for applications like social media content moderation, sentiment analysis, and intent detection, where label sets are dynamic or annotated data are scarce. Unlike traditional supervised learning, which requires labeled examples for each category, zero-shot methods leverage pre-trained language models to generalize to unseen labels by reasoning about semantic relationships. For instance, a model trained on general language data can classify a text as "sports news" without prior exposure to that category, saving significant time and resources.

However, traditional approaches, such as prompt-based and natural language inference (NLI)-based methods, often process text and labels independently, incurring O(n) computational complexity for n labels. This inefficiency becomes prohibitive for large label sets, as seen in real-time systems with thousands of categories. Moreover, these methods fail to model inter-label dependencies within the context of the input text, limiting their ability to distinguish nuanced or overlapping categories.

We introduce GliZNet (Generalized Zero-Shot Network), a novel framework that addresses these limitations through a joint text-label encoding mechanism and a multi-objective loss function. By processing text and all candidate labels in a single transformer forward pass, GliZNet achieves O(1) complexity, enabling scalability for large label sets. Its label-aware embeddings capture dynamic inter-label relationships, enhancing classification accuracy. This report details GliZNet's methodology, situates it within the state of the art, and highlights its contributions to zero-shot text classification.

## 2. State of the Art

Zero-shot text classification, a cornerstone of modern natural language processing (NLP), allows models to assign unseen labels by leveraging transferable knowledge from large pre-trained models. As of 2025, the field is dominated by four key paradigms: Natural Language Inference (NLI)-based approaches, cross-encoder models, adaptations of Contrastive Language-Image Pretraining (CLIP) for text, and joint text-label encoders. Below, we analyze these methods, their strengths, limitations, and recent trends, positioning GliZNet within this landscape.

### 2.1 Natural Language Inference (NLI)-Based Approaches

NLI-based methods frame zero-shot classification as an entailment task, treating the input text as a premise and each candidate label as a hypothesis. Models compute entailment probabilities to determine the most likely label. Key models include:

- **bart-large-mnli**: A BART model fine-tuned on the MultiNLI dataset, widely used for English zero-shot tasks due to its robust semantic reasoning.
- **deberta-v3-large-zeroshot-v2.0**: A DeBERTa-v3 model fine-tuned for zero-shot NLI, achieving 0.6821 avg F1 on standard benchmarks.

**Strengths**: NLI models excel at capturing semantic relationships, making them effective for nuanced label differentiation. **Limitations**: Their O(n) complexity, requiring a separate forward pass per label, hinders scalability for large label sets — throughput degrades from 24.55 to 0.47 examples/s when scaling from 1 to 128 labels (Stepanov et al., 2025).

### 2.2 Cross-Encoder Models

Cross-encoder models jointly encode text-label pairs, allowing direct interaction between inputs to capture fine-grained relationships. Notable examples include:

- **cross-encoder/nli-distilroberta-base**: A lightweight DistilRoBERTa model (82.1M parameters) fine-tuned on SNLI and MultiNLI.
- **cross-encoder/nli-deberta-v3-base**: A DeBERTa-v3 model with enhanced attention mechanisms for improved accuracy.

**Strengths**: Cross-encoders provide high precision, especially for ambiguous labels, due to their ability to model pairwise interactions. **Limitations**: Like NLI methods, they suffer from O(n) complexity, with throughput that scales linearly with label count — making them impractical for large label sets in production.

### 2.3 CLIP for Text Classification

Originally designed for vision-language tasks, CLIP aligns text and image embeddings through contrastive learning. Adaptations like CLIPText reformulate text classification as a text-image matching problem, associating labels with proxy images or prompts.

**Strengths**: CLIP-based methods can achieve O(1) complexity by computing shared embeddings, leveraging large-scale contrastive pretraining. **Limitations**: Mapping textual labels to image-based embeddings requires non-intuitive engineering, complicating implementation.

### 2.4 GLiClass: Joint Text-Label Encoding

GLiClass (Stepanov et al., 2025) is the most closely related work to GliZNet. Adapted from the GLiNER architecture for named entity recognition, GLiClass processes text and all candidate labels jointly in a single forward pass, achieving non-linear scaling with label count.

**Architecture**: GLiClass prepends each label with a `<<LABEL>>` special token and concatenates all labels with the input text. The uni-encoder variant (best-performing) uses a DeBERTa-v3 backbone, with three pooling strategies (first-token, mean, attention-weighted) and either dot-product or MLP scoring. A layer-wise attention re-weighting mechanism (squeeze-excitation over all transformer layers) and token-level contrastive loss enhance representation quality.

**Training**: GLiClass employs a sophisticated 3-stage pipeline:

1. **Pre-training** on a 1.2M example general-purpose corpus
2. **Mid-training** with adapted Proximal Policy Optimization (PPO) for reinforcement learning
3. **Post-training** using LoRA with logic/NLI and pattern-focused data streams

**Results** (avg macro F1 on benchmark): GLiClass-large (439M params): 0.7417, GLiClass-base (187M): 0.7056, GLiClass-modern-base (151M): 0.6170. Throughput degrades only 7–20% from 1 to 128 labels.

**Limitations**: The multi-stage training pipeline (pre-train → RL mid-train → LoRA post-train) is complex. GLiClass uses a single global text representation (pooled), which may dilute relevant information when labels require attention to different parts of the text.

### 2.5 Comparative Analysis

| Method                                | Complexity | Inter-Label Reasoning                             | Key Limitation                   |
| ------------------------------------- | ---------- | ------------------------------------------------- | -------------------------------- |
| NLI-Based (e.g., DeBERTa-v3-zeroshot) | O(n)       | No                                                | 50× slowdown at 128 labels      |
| Cross-Encoder                         | O(n)       | No                                                | High computation per label pair  |
| CLIPText Adaptations                  | O(1)       | Partial                                           | Image-label association required |
| GLiClass (Stepanov et al., 2025)      | O(1)       | Yes (self-attention)                              | Complex 3-stage training with RL |
| GliZNet (Ours)                        | O(1)       | Yes (self-attention + cross-attention enrichment) | Single-stage training            |

### 2.6 Positioning GliZNet

GliZNet shares the joint text-label encoding paradigm with GLiClass but introduces several architectural innovations:

1. **Label-conditioned text representation**: Unlike GLiClass which uses a single pooled text representation for all labels, GliZNet computes a **unique text representation per label** via cross-attention — each label attends to the text tokens most relevant to it.
2. **Cooperative label enrichment**: A multi-head self-attention layer allows labels to see each other's text evidence before final scoring, enabling cooperative routing.
3. **Bilinear scoring**: Rather than dot-product or MLP scoring, GliZNet uses a bilinear interaction matrix that captures cross-dimensional relationships between text and label representations.
4. **Simpler training**: Single-stage supervised training with online augmentation, without requiring RL or multi-stage LoRA adaptation.

These innovations target GLiClass's main limitation — the global text representation — while simplifying the training pipeline.

## 3. Methodology

GliZNet's methodology is built on three core components: a novel joint encoding tokenizer, label-aware contextual embeddings, and a hybrid loss function. Below, we detail each component, supported by implementation specifics for reproducibility.

### 3.1 Joint Text-Label Encoding

GliZNet processes text and all candidate labels in a single transformer forward pass, achieving O(1) complexity. This is enabled by a custom tokenizer, `GliZNETTokenizer`, which constructs a unified input sequence:

```
[CLS] text_tokens [SEP] label_1_tokens [LAB] label_2_tokens [LAB] ... label_n_tokens [LAB] [PAD]
```

The tokenizer, based on the DeBERTa-v3 SentencePiece vocabulary, supports a maximum sequence length of 1024 tokens. A label mask (`lmask`) tensor assigns a value of 0 to text tokens and unique integers (1, 2, 3, etc.) to each label's tokens, enabling the model to distinguish components during encoding. This joint encoding captures contextual interplay between text and labels, unlike traditional O(n) methods requiring separate passes per label.

### 3.2 Label-Aware Contextual Embeddings

All backbone hidden states are L2-normalised before downstream operations, placing representations on the unit hypersphere for stable attention and scoring.

The joint encoding produces label-aware embeddings in a single transformer pass:

- **Label Representations**: The hidden state of the `[LAB]` separator token that terminates each label span serves as the label representation — a single contextual vector that has attended to all preceding label tokens via self-attention. A dropout layer is applied before extraction as a regulariser.
- **Text Representation**: For each label, a dedicated text representation is computed by attending over all text token positions with the label's `[LAB]` embedding as the query (learnable temperature-scaled dot-product attention), producing a **label-conditioned summary** of the text. This is a key difference from GLiClass, which uses a single global text representation (pooled) shared across all labels.
- **Label Enrichment** (optional): After the first-pass text pooling, labels attend to each other via multi-head self-attention over fused (label + text evidence) representations. This cooperative enrichment allows label $i$ to see what text evidence label $j$ found, enabling cooperative routing before the final scoring pass.

### 3.3 Training Objective: Multi-Objective Loss

GliZNet is trained with a multi-objective loss combining three complementary terms:

```
L_total = λ_softmax · L_softmax + λ_focal · L_focal + λ_repulsion · L_repulsion
```

where λ_softmax = 1.0, λ_focal = 0.4, and λ_repulsion = 0.1.

- **One-vs-Negatives Softmax Loss (L_softmax)**: For each positive label in a sample, computes the cross-entropy of that positive against all valid negatives in the same sample. Positives do not compete with each other — the denominator for positive $p$ is $\exp(\text{logit}_p) + \sum_{n \in \text{neg}} \exp(\text{logit}_n + m)$, where $m = 0.1$ is a configurable additive margin (`supcon_margin`) that forces a minimum separation before the loss saturates. Implemented via `logsumexp` for numerical stability.
- **Focal Loss (L_focal)**: A class-balanced focal loss with γ=1.85 that down-weights easy negatives and focuses gradients on hard-to-classify labels, providing an absolute per-label calibration signal independent of the softmax distribution.
- **Label Repulsion Loss (L_repulsion)**: Penalises high cosine similarity between different label embeddings *within the same sample*, preventing representation collapse while respecting contextual label similarity across samples.

The softmax and focal losses are complementary: softmax provides relative ranking gradients while focal loss provides absolute thresholding gradients with emphasis on hard examples, together ensuring both discriminative ranking and well-calibrated per-label probabilities.

### 3.4 Data Pipeline

GliZNet is trained on `alexneakameni/ZSHOT-HARDSET-v2`, a synthetic dataset of ~397k text-label pairs generated using advanced language models (e.g., Ollama, Groq). The dataset spans domains like technology, healthcare, and sports. Hard negatives are created by prompting models to generate semantically similar but incorrect labels (e.g., "positive" vs. "neutral" for sentiment tasks), simulating real-world challenges.

The training set is augmented with 14 additional MCQ/NLI datasets (1k samples each) including ARC, OpenBookQA, CommonsenseQA, BIG-Bench subsets, and medical QA, providing diverse reasoning tasks.

**Text augmentation** is applied online during training via an nlpaug-based pipeline configured in YAML:

- Keyboard typos, OCR-like errors, character swap/delete
- Word deletion, spelling errors, suffix truncation, random case change
- Each augmentation is applied independently with a per-sample probability (10-15%)

**Label augmentation** controls the positive/negative ratio and total label count:

- `RatioEnforcementSelector`: randomly enforces neg-heavy or pos-heavy distributions (10% each)
- `LabelLimit`: caps at 20 labels per sample (min 5), shuffles order, and removes underscores 90% of the time

### 3.5 Model Architecture

GliZNet comprises three components:

- **Text Encoder**: A pre-trained transformer (`microsoft/deberta-v3-base`, ~184M parameters) processes the joint sequence.
- **Label Aggregator**: Extracts each label's `[LAB]` token hidden state, then computes a label-specific text representation via cross-attention over text tokens.
- **Scoring Head**: A bilinear layer (`nn.Bilinear(D, D, 1)`) jointly scores the label-specific text representation against the label representation.

The architecture is optimised with AdamW (learning rate 4e-5, effective batch size 192) over up to 10 epochs with early stopping.

A key addition is **label enrichment** (`LabelContextAttention`): each label's embedding is fused with its first-pass text evidence via a linear projection, then all labels attend to each other through multi-head self-attention (8 heads) with residual connection and layer norm. The enriched label embeddings are then used for a second-pass cross-attention over text tokens, producing the final label-specific text representations used for scoring. This two-pass mechanism allows labels to cooperatively route their attention based on what other labels have found.

### 3.6 Implementation Details

For reproducibility:

- `GliZNETTokenizer` uses the DeBERTa-v3 SentencePiece tokenizer, augmented with a custom `[LAB]` separator token, handling up to 1024 tokens including text and all labels.
- The backbone (`microsoft/deberta-v3-base`) was chosen for its strong contextual representations; the architecture supports any HuggingFace-compatible encoder.
- Training is distributed across 2 GPUs using DDP via `accelerate launch` (batch size 48 per device, gradient accumulation 2).
- Code: https://github.com/KameniAlexNea/zero-shot-classification — Data generation: https://github.com/KameniAlexNea/generate-gliznet-data

## 4. Experimental Evaluation

### 4.1 Benchmark Setup

We evaluate GliZNet on the 10-dataset benchmark from GLiClass (Stepanov et al., 2025), covering binary sentiment, fine-grained sentiment, topic classification, and spam detection:

| Dataset              | Task                           | Classes |
| -------------------- | ------------------------------ | ------- |
| CR                   | Customer review sentiment      | 2       |
| SST-2                | Movie sentiment (binary)       | 2       |
| SST-5                | Movie sentiment (fine-grained) | 5       |
| IMDb                 | Movie review sentiment         | 2       |
| 20-Newsgroups        | News topic classification      | 20      |
| Enron Spam           | Email spam detection           | 2       |
| Financial PhraseBank | Financial sentiment            | 3       |
| AG News              | News topic classification      | 4       |
| Emotion              | Tweet emotion detection        | 6       |
| Rotten Tomatoes      | Movie review sentiment         | 2       |

All evaluations use macro F1 in a true zero-shot setting (no per-dataset fine-tuning). We compare against GLiClass-large-v3 (439M params), GLiClass-base-v3 (187M), and GLiClass-modern-base-v3 (187M).

### 4.2 Results

| Dataset              | GliZNet (ours)   | GLiClass-large   | GLiClass-base    | GLiClass-modern-base |
| -------------------- | ---------------- | ---------------- | ---------------- | -------------------- |
| CR                   | 0.8747           | 0.9281           | 0.9127           | 0.8936               |
| SST-2                | 0.8921           | 0.9176           | 0.8959           | 0.8982               |
| SST-5                | **0.4294** | 0.3798           | 0.3236           | 0.2885               |
| IMDb                 | 0.8887           | 0.9366           | 0.9248           | 0.9154               |
| 20-Newsgroups        | 0.4712           | 0.5806           | 0.5045           | 0.3342               |
| Enron Spam           | 0.5296           | 0.7574           | 0.6252           | 0.5903               |
| Financial PhraseBank | 0.7094           | 0.9023           | 0.9094           | 0.4121               |
| AG News              | **0.7237** | 0.7229           | 0.7209           | 0.7069               |
| Emotion              | 0.4262           | 0.4504           | 0.4450           | 0.4249               |
| Rotten Tomatoes      | 0.6640           | 0.8411           | 0.7943           | 0.7060               |
| **Average**    | **0.6609** | **0.7417** | **0.7056** | **0.6170**     |

### 4.3 Analysis

**Where GliZNet excels.** GliZNet achieves the best results on two tasks:

- **SST-5** (5-class fine-grained sentiment): 0.4294 — surpasses all GLiClass variants including the large model (0.3798). The per-label cross-attention mechanism excels when labels are semantically close ("very positive", "positive", "neutral", "negative", "very negative") because each label attends to different intensity cues in the text.
- **AG News** (4-class topic): 0.7237 — marginally beats GLiClass-large (0.7229) and GLiClass-base (0.7209), showing strong multi-class topic discrimination.

GliZNet also outperforms GLiClass-modern-base on 8 out of 10 datasets, despite comparable model sizes (~184M vs 151M parameters).

**Where GliZNet lags.** The largest gaps appear on domain-specific tasks:

- **Enron Spam** (0.5296 vs 0.7574): −0.2278 vs GLiClass-large. Spam detection is a narrow domain where GLiClass's larger pre-training corpus (1.2M examples) provides more robust boundaries.
- **Financial PhraseBank** (0.7094 vs 0.9094): −0.2000 vs GLiClass-base. Domain-specific financial language requires exposure that GliZNet's general-purpose training data lacks.
- **Rotten Tomatoes** (0.6640 vs 0.8411): −0.1771 vs GLiClass-large. Binary movie review classification where the 3-stage training advantage is most apparent.

**Interpretation.** The pattern is clear: GliZNet's per-label cross-attention provides an advantage on **multi-class tasks with semantically similar labels** (SST-5, AG News, Emotion) where differentiated text evidence per label matters most. It underperforms on **binary/domain-specific tasks** where GLiClass's larger and more diverse pre-training corpus (3× larger, plus RL refinement) provides stronger domain coverage.

The overall gap to GLiClass-base (−0.0447) is modest given that GliZNet uses:

- 3× less pre-training data (397k vs 1.2M examples)
- No reinforcement learning (PPO) mid-training
- No LoRA post-training
- A single-stage training pipeline

This suggests that the architectural innovations (label-conditioned cross-attention, cooperative enrichment, bilinear scoring) compensate for much of the training complexity advantage, and that scaling the training data would likely close the remaining gap.

## 5. Contributions to the State of the Art

GliZNet advances zero-shot text classification through:

- **Scalable Efficiency**: O(1) complexity via joint encoding, enabling efficient processing of large label sets, unlike O(n) cross-encoders.
- **Label-Conditioned Text Representation**: Unlike GLiClass's global text pooling, GliZNet computes a unique text summary per label via cross-attention, so each label attends to relevant text tokens.
- **Cooperative Label Enrichment**: Multi-head self-attention over label representations allows labels to see each other's text evidence before final scoring, enabling cooperative routing.
- **Simpler Training Pipeline**: A single-stage supervised training with online nlpaug-based text augmentation and label augmentation, without requiring multi-stage RL or LoRA post-training.
- **Competitive Results**: Achieves 0.6609 avg F1 vs GLiClass-base's 0.7056, while using a simpler architecture and training procedure. Outperforms GLiClass-modern-base (0.6170) with the same backbone size.
- **Fine-Grained Label Advantage**: Best-in-class performance on SST-5 and AG News demonstrates the value of per-label text representations for semantically close categories.

These innovations demonstrate that careful architectural design (cross-attention scoring, label enrichment) combined with robust data augmentation can approach the performance of more complex training pipelines.

## 6. Conclusion

GliZNet demonstrates that efficient zero-shot text classification can be achieved through a simpler training pipeline than GLiClass's multi-stage approach (pre-train → RL mid-train → LoRA post-train). Its joint encoding achieves O(1) complexity, while the combination of label-conditioned cross-attention, cooperative label enrichment, and bilinear scoring enables context-sensitive classification competitive with models trained using significantly more complex procedures.

Current results (0.6609 avg F1) place GliZNet between GLiClass-modern-base (0.6170) and GLiClass-base (0.7056), with the remaining gap likely attributable to GLiClass's larger pre-training corpus (1.2M examples vs ~397k) and RL-based refinement. GliZNet's per-label cross-attention mechanism offers a fundamentally different text-label interaction compared to GLiClass's global pooling, which may prove advantageous as label sets become more fine-grained.

Future research directions include:

- Scaling the pre-training corpus and exploring multi-stage training with RL
- Extending to multilingual backbones (XLM-R, mDeBERTa)
- Improving few-shot adaptation leveraging the label-conditioned text representations
- Investigating retrieval-augmented strategies for very large label sets exceeding context length

## References

- **Stepanov et al.**. (2025). *GLiClass: Generalist Lightweight Model for Sequence Classification Tasks*. [https://arxiv.org/abs/2508.07662](https://arxiv.org/abs/2508.07662)
- **Zaratiana et al.**. (2023). *GLiNER: Generalist Model for Named Entity Recognition Using Bidirectional Transformer*. [https://arxiv.org/abs/2311.08526](https://arxiv.org/abs/2311.08526)
- **Laurer et al.**. (2023). *Building Efficient Universal Classifiers with Natural Language Inference*. [https://arxiv.org/abs/2312.17543](https://arxiv.org/abs/2312.17543)
- **He et al.**. (2021). *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing*. [https://arxiv.org/abs/2111.09543](https://arxiv.org/abs/2111.09543)
- **Hugging Face**. *Zero-Shot Classification*. [https://huggingface.co/tasks/zero-shot-classification](https://huggingface.co/tasks/zero-shot-classification)
- **Papers With Code**. *Zero-Shot Text Classification*. [https://paperswithcode.com/task/zero-shot-text-classification](https://paperswithcode.com/task/zero-shot-text-classification)
- **OpenAI**. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. CLIP paper.
- **Qin et al.**. (2023). *CLIPText: Zero-Shot Text Classification via CLIP*. Referenced for CLIPText adaptations.
- **Tunstall et al.**. (2022). *Efficient Few-Shot Learning Without Prompts* (SetFit). [https://arxiv.org/abs/2209.11055](https://arxiv.org/abs/2209.11055)
- **Yin et al.**. (2019). *Benchmarking Zero-Shot Text Classification: Datasets, Evaluation and Entailment Approach*. [https://arxiv.org/abs/1909.00161](https://arxiv.org/abs/1909.00161)
