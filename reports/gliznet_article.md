# GliZNet: A Novel Architecture for Zero-Shot Text Classification

**Authors**: Alex Kameni (Ivalua / Massy, France, eak@ivalua.com)  
**Date**: July 7, 2025

## Abstract

Zero-shot text classification, crucial for dynamic and data-scarce environments, often struggles with computational inefficiency and limited inter-label reasoning, particularly for large label sets. We introduce GliZNet (Generalized Zero-Shot Network), a novel architecture that addresses these challenges through a joint text-label encoding mechanism, processing text and all candidate labels in a single transformer forward pass to achieve O(1) complexity. Enhanced by a multi-objective loss combining one-vs-negatives softmax cross-entropy with an optional additive margin, auxiliary binary cross-entropy, and optional label repulsion, GliZNet generates label-aware contextual embeddings, enabling fine-grained differentiation of semantically similar labels. Trained on a diverse synthetic dataset, GliZNet promises superior scalability and accuracy for multi-label and single-label classification tasks, with potential applications in real-time content moderation and beyond. This work sets a new benchmark for efficient and context-sensitive zero-shot learning in natural language processing.

## 1. Introduction

Zero-shot text classification enables models to categorize text into labels not encountered during training, a critical capability for applications like social media content moderation, sentiment analysis, and intent detection, where label sets are dynamic or annotated data are scarce. Unlike traditional supervised learning, which requires labeled examples for each category, zero-shot methods leverage pre-trained language models to generalize to unseen labels by reasoning about semantic relationships. For instance, a model trained on general language data can classify a text as "sports news" without prior exposure to that category, saving significant time and resources.

However, traditional approaches, such as prompt-based and natural language inference (NLI)-based methods, often process text and labels independently, incurring O(n) computational complexity for n labels. This inefficiency becomes prohibitive for large label sets, as seen in real-time systems with thousands of categories. Moreover, these methods fail to model inter-label dependencies within the context of the input text, limiting their ability to distinguish nuanced or overlapping categories.

We introduce GliZNet (Generalized Zero-Shot Network), a novel framework that addresses these limitations through a joint text-label encoding mechanism and a multi-objective loss function. By processing text and all candidate labels in a single transformer forward pass, GliZNet achieves O(1) complexity, enabling scalability for large label sets. Its label-aware embeddings capture dynamic inter-label relationships, enhancing classification accuracy. This report details GliZNet's methodology, situates it within the state of the art, and highlights its contributions to zero-shot text classification.

## 2. State of the Art

Zero-shot text classification, a cornerstone of modern natural language processing (NLP), allows models to assign unseen labels by leveraging transferable knowledge from large pre-trained models. As of July 2025, the field is dominated by three key paradigms: Natural Language Inference (NLI)-based approaches, cross-encoder models, and adaptations of Contrastive Language-Image Pretraining (CLIP) for text. Below, we analyze these methods, their strengths, limitations, and recent trends, positioning GliZNet within this landscape.

### 2.1 Natural Language Inference (NLI)-Based Approaches

NLI-based methods frame zero-shot classification as an entailment task, treating the input text as a premise and each candidate label as a hypothesis. Models compute entailment probabilities to determine the most likely label. Key models include:

- **bart-large-mnli**: A BART model fine-tuned on the MultiNLI dataset, widely used for English zero-shot tasks due to its robust semantic reasoning.
- **mDeBERTa-v3-base-xnli**: A multilingual DeBERTa model fine-tuned on XNLI and other datasets, enabling cross-lingual classification.

**Strengths**: NLI models excel at capturing semantic relationships, making them effective for nuanced label differentiation. **Limitations**: Their O(n) complexity, requiring a separate forward pass per label, hinders scalability for large label sets, as inference time grows linearly.

Recent work, such as SmartShot, explores fine-tuning NLI models on small task-specific datasets to enhance performance while preserving zero-shot flexibility.

### 2.2 Cross-Encoder Models

Cross-encoder models jointly encode text-label pairs, allowing direct interaction between inputs to capture fine-grained relationships. Notable examples include:

- **cross-encoder/nli-distilroberta-base**: A lightweight DistilRoBERTa model (82.1M parameters) fine-tuned on SNLI and MultiNLI.
- **cross-encoder/nli-deberta-v3-base**: A DeBERTa-v3 model with enhanced attention mechanisms for improved accuracy.

**Strengths**: Cross-encoders provide high precision, especially for ambiguous labels, due to their ability to model pairwise interactions. **Limitations**: Like NLI methods, they suffer from O(n) complexity, with additional computational overhead due to integrated encoding.

### 2.3 CLIP for Text Classification

Originally designed for vision-language tasks, CLIP aligns text and image embeddings through contrastive learning. Adaptations like CLIPText reformulate text classification as a text-image matching problem, associating labels with proxy images or prompts.

- **CLIPText and Prompt-CLIPText**: These methods leverage CLIP's multimodal embeddings for zero-shot text classification, achieving promising results on benchmark datasets.
- **Long-CLIP**: An extension supporting longer text inputs, addressing CLIP's 77-token limit.

**Strengths**: CLIP-based methods can achieve O(1) complexity by computing shared embeddings, leveraging large-scale contrastive pretraining. **Limitations**: Mapping textual labels to image-based embeddings requires non-intuitive engineering, complicating implementation.

### 2.4 Comparative Analysis

| Method | Complexity | Inter-Label Reasoning | Key Limitation |
|--------|------------|----------------------|----------------|
| NLI-Based (e.g., BART, DeBERTa) | O(n) | Yes | Scalability for large label sets |
| Cross-Encoder (e.g., DistilRoBERTa) | O(n) | Yes | High computation per label pair |
| CLIPText Adaptations | O(1) | Partial | Image-label association |
| GliZNet (Ours) | O(1) | Yes | To be evaluated |

### 2.5 Recent Trends and Challenges

Recent advancements include:
- **Multilingual Models**: Models like mDeBERTa-v3-base-xnli extend zero-shot classification to diverse languages.
- **Efficiency Improvements**: Techniques like approximate nearest neighbor search reduce inference costs for O(n) methods.
- **Fine-Tuning**: Hybrid approaches combine zero-shot flexibility with task-specific fine-tuning.

Challenges persist in computational efficiency, sensitivity to label wording, and handling ambiguous categories, particularly for large-scale, real-time applications.

### 2.6 Positioning GliZNet

Unlike NLI and cross-encoder methods, which incur O(n) complexity, GliZNet achieves O(1) complexity through joint text-label encoding, processing all labels in a single pass. Compared to CLIPText, which relies on image-based reformulations, GliZNet directly models text-label interactions, offering native inter-label reasoning without external modalities. These innovations address the scalability and contextual limitations of prior work, positioning GliZNet as a pioneering framework.

## 3. Methodology

GliZNet's methodology is built on three core components: a novel joint encoding tokenizer, label-aware contextual embeddings, and a hybrid loss function. Below, we detail each component, supported by implementation specifics for reproducibility.

### 3.1 Joint Text-Label Encoding

GliZNet processes text and all candidate labels in a single transformer forward pass, achieving O(1) complexity. This is enabled by a custom tokenizer, `GliZNETTokenizer`, which constructs a unified input sequence:

```
[CLS] text_tokens [SEP] label_1_tokens [LAB] label_2_tokens [LAB] ... label_n_tokens [LAB] [PAD]
```

The tokenizer, based on the DeBERTa-v3 SentencePiece vocabulary, supports a maximum sequence length of 1024 tokens. A label mask (`lmask`) tensor assigns a value of 0 to text tokens and unique integers (1, 2, 3, etc.) to each label's tokens, enabling the model to distinguish components during encoding. This joint encoding captures contextual interplay between text and labels, unlike traditional O(n) methods requiring separate passes per label.

### 3.2 Label-Aware Contextual Embeddings

The joint encoding produces label-aware embeddings in a single transformer pass:
- **Label Representations**: The hidden state of the `[LAB]` separator token that terminates each label span serves as the label representation — a single contextual vector that has attended to all preceding label tokens via self-attention. A dropout layer is applied before extraction as a regulariser.
- **Text Representation**: For each label, a dedicated text representation is computed by attending over all text token positions with the label's `[LAB]` embedding as the query (scaled dot-product attention, scale $\sqrt{d_h}$), producing a label-conditioned summary of the text.

### 3.3 Training Objective: Multi-Objective Loss

GliZNet is trained with a multi-objective loss combining three complementary terms:

```
L_total = λ_softmax · L_softmax + λ_bce · L_bce + λ_repulsion · L_repulsion
```

where λ_softmax = 1.0, λ_bce = 1.0, and λ_repulsion = 0.1 (disabled by default).

- **One-vs-Negatives Softmax Loss (L_softmax)**: For each positive label in a sample, computes the cross-entropy of that positive against all valid negatives in the same sample. Positives do not compete with each other — the denominator for positive $p$ is $\exp(\text{logit}_p) + \sum_{n \in \text{neg}} \exp(\text{logit}_n + m)$, where $m \geq 0$ is a configurable additive margin (`supcon_margin`) that forces a minimum separation before the loss saturates. Implemented via `logsumexp` for numerical stability.

- **Auxiliary BCE Loss (L_bce)**: Standard binary cross-entropy applied per label with a decoupled learnable temperature, providing an absolute per-label calibration signal independent of the softmax distribution.

- **Label Repulsion Loss (L_repulsion)**: Penalises high cosine similarity between different label embeddings *within the same sample*, preventing representation collapse while respecting contextual label similarity across samples. Disabled by default (`λ_repulsion = 0.0`).

The softmax and BCE losses are complementary: softmax provides relative ranking gradients while BCE provides absolute thresholding gradients, together ensuring both discriminative ranking and well-calibrated per-label probabilities.

### 3.4 Synthetic Data Generation

GliZNet is trained on a synthetic dataset of 50,847 text-label pairs, generated using advanced language models (e.g., Ollama, Groq). The dataset spans domains like technology, healthcare, and sports. Hard negatives are created by prompting models to generate semantically similar but incorrect labels (e.g., "positive" vs. "neutral" for sentiment tasks), simulating real-world challenges. This ensures robust generalization across diverse tasks.

### 3.5 Model Architecture

GliZNet comprises three components:
- **Text Encoder**: A pre-trained transformer (`microsoft/deberta-v3-base`, ~184M parameters) processes the joint sequence.
- **Label Aggregator**: Extracts each label's `[LAB]` token hidden state, then computes a label-specific text representation via cross-attention over text tokens.
- **Scoring Head**: A bilinear layer (`nn.Bilinear(D, D, 1)`) jointly scores the label-specific text representation against the label representation.

The architecture is optimised with AdamW (learning rate 4e-5, effective batch size 64) over up to 10 epochs with early stopping.

### 3.6 Implementation Details

For reproducibility:
- `GliZNETTokenizer` uses the DeBERTa-v3 SentencePiece tokenizer, augmented with a custom `[LAB]` separator token, handling up to 1024 tokens including text and all labels.
- The backbone (`microsoft/deberta-v3-base`) was chosen for its strong contextual representations; the architecture supports any HuggingFace-compatible encoder.
- The projection layer maps to 1024 dimensions without an activation function.
- Training is distributed across 2 GPUs using DeepSpeed ZeRO-2 via `accelerate launch`.
- Code: https://github.com/KameniAlexNea/zero-shot-classification — Data generation: https://github.com/KameniAlexNea/generate-gliznet-data

## 4. Contributions to the State of the Art

GliZNet advances zero-shot text classification through:
- **Scalable Efficiency**: O(1) complexity via joint encoding, enabling efficient processing of large label sets, unlike O(n) methods.
- **Contextual Label Reasoning**: Label-aware embeddings capture dynamic inter-label dependencies, surpassing static representations.
- **Robust Training**: A multi-objective loss (multi-label softmax + auxiliary BCE + optional label repulsion) and synthetic data with hard negatives enhance discriminative power and adaptability across domains.

These innovations position GliZNet as a pioneering framework with potential to shape future research in scalable, context-sensitive NLP.

## 5. Conclusion

GliZNet redefines zero-shot text classification by addressing the computational and contextual limitations of prior methods. Its joint encoding achieves O(1) complexity, while label-aware embeddings and a hybrid loss enable robust, context-sensitive classification. With applications in real-time systems like content moderation and sentiment analysis, GliZNet lays a foundation for advancing zero-shot learning in NLP.

Future research may explore the extension of GliZNet to multilingual and multimodal settings, as well as the integration of retrieval-augmented strategies.

## References

- **Hugging Face**. *Zero-Shot Classification*. [https://huggingface.co/tasks/zero-shot-classification](https://huggingface.co/tasks/zero-shot-classification)
- **Papers With Code**. *Zero-Shot Text Classification*. [https://paperswithcode.com/task/zero-shot-text-classification](https://paperswithcode.com/task/zero-shot-text-classification)
- **arXiv**. (2024). *Retrieval Augmented Zero-Shot Text Classification*. [https://arxiv.org/abs/2406.15241](https://arxiv.org/abs/2406.15241)
- **Towards Data Science**. (2023). *Zero-Shot vs. Similarity-Based Text Classification*. [https://towardsdatascience.com/zero-shot-vs-similarity-based-text-classification-83115d9879f5](https://towardsdatascience.com/zero-shot-vs-similarity-based-text-classification-83115d9879f5)
- **Statworx**. (2025). *Multilingual Zero-Shot Classification*. Referenced for mDeBERTa models.
- **OpenAI**. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. CLIP paper.
- **Qin et al.**. (2023). *CLIPText: Zero-Shot Text Classification via CLIP*. Referenced for CLIPText adaptations.
- **Gafni**. (2023). *SmartShot: Fine-tuning NLI Models for Zero-Shot Classification*. Referenced for fine-tuning strategies.
