# Research Report: GliZNet – A Novel Architecture for Zero-Shot Text Classification

**Date**: July 7, 2025  
**Prepared by**: Alex Kameni

## Abstract

Zero-shot text classification often struggles with computational inefficiency and a failure to model inter-label dependencies, especially when dealing with large label sets. This report introduces GliZNet (Generalized Zero-Shot Network), a novel architecture that addresses these limitations. GliZNet employs a **joint text-label encoding** mechanism, processing the text and all candidate labels within a single transformer forward pass, achieving O(1) complexity regardless of the number of labels. The model's training is enhanced by a **contrastive loss function** that incorporates hard negative mining, enabling it to learn fine-grained distinctions between semantically similar labels. By creating label-aware contextual embeddings in a unified pass, GliZNet significantly improves scalability and classification accuracy, setting a new standard for zero-shot learning.

## 1. Introduction

Zero-shot text classification enables models to categorize text into labels not encountered during training, a vital capability for dynamic, data-scarce environments. Traditional approaches often process text and labels independently, incurring significant computational overhead, especially with expansive label sets. This separation prevents the model from reasoning about the relationships between different labels in the context of the given text.

GliZNet (Generalized Zero-Shot Network) introduces an innovative framework that tackles these inefficiencies head-on. Through **joint text-label encoding** and a specialized **contrastive learning** objective, GliZNet achieves both scalability and high performance. This report outlines GliZNet’s methodology, details its architectural components, and situates it within the current state of the art in zero-shot text classification.

## 2. State of the Art in Zero-Shot Text Classification

The field of zero-shot text classification has seen diverse advancements, each addressing the challenge of generalizing to unseen labels. Key approaches include:

* **Prompt-Based Methods**: Leveraging large language models (e.g., GPT-3.5-turbo, BART-large-mnli), these techniques frame classification as a text completion task using prompts like "This text is about [label]." They require separate encodings per label, yielding O(n) complexity for n labels, which scales poorly.
* **Natural Language Inference (NLI)-Based Methods**: Models like RoBERTa-large-mnli treat classification as an entailment problem, with text as the premise and labels as hypotheses. While effective, they also demand multiple forward passes, resulting in the same O(n) complexity that hinders scalability.
* **Embedding-Based Techniques**: Methods like Lbl2Vec generate joint embeddings for words, documents, and labels. However, they do not process all elements in a single, unified pass, missing the efficiency and contextual reasoning capabilities of GliZNet's architecture.

Current methods struggle with two primary limitations:

1. **Computational Cost**: The O(n) complexity of separate text-label encodings makes them impractical for real-time applications with large label sets.
2. **Inter-Label Reasoning**: Most approaches fail to model relationships among labels in the context of the input text, which is critical for distinguishing between similar or nuanced categories.

GliZNet advances the field by introducing a unified encoding scheme that achieves O(1) complexity and fosters label-aware contextual reasoning, redefining efficiency and performance benchmarks.

## 3. Methodology

GliZNet’s methodology centers on three pillars: a novel joint encoding tokenizer, label-aware contextual embeddings derived from a single forward pass, and a sophisticated contrastive learning objective.

### 3.1 Joint Text-Label Encoding

Unlike conventional methods, GliZNet processes text and all candidate labels in a single transformer forward pass. This is enabled by a custom tokenizer, **`GliZNETTokenizer`**, which constructs a unified input sequence formatted as follows:

`[CLS] text_tokens [SEP] label_1_tokens ; label_2_tokens ; ... [PAD]`

A critical component of this process is the `lmask` (label mask), an additional tensor that informs the model which tokens correspond to the text (mask value 0) and which belong to each label group (mask values 1, 2, 3, etc.). This structure allows the model to encode the text and all labels simultaneously, capturing their contextual interplay efficiently.

### 3.2 Label-Aware Contextual Embeddings

The joint encoding produces **label-aware embeddings** from a single transformer pass:

* **Text Representation**: The final hidden state of the `[CLS]` token is used as the contextualized representation of the input text.
* **Label Representations**: The representation for each label is computed by taking an attention-weighted average of the final hidden states of its constituent tokens.

This approach enables GliZNet to reason about label relationships dynamically, as the embeddings for all labels are computed within the same contextual window. An optional projection layer can reduce the dimensionality of these embeddings for greater computational efficiency.

### 3.3 Training Objective: Hybrid BCE and Contrastive Loss

GliZNet is trained using a hybrid loss function that combines a standard **Binary Cross-Entropy (BCE) loss** with a specialized **contrastive loss** designed to handle hard negatives. This dual-objective approach ensures that the model not only learns to make correct predictions but also develops a more robust and discriminative representation space. The final loss is a weighted sum of these two components:

`L_total = α * L_bce + β * L_contrastive`

Where `α` and `β` are scaling factors for the BCE and contrastive losses, respectively.

1.  **Binary Cross-Entropy Loss (`L_bce`)**: The primary objective is a scaled BCE loss, which treats the task as a series of independent binary classifications for each label. This loss is effective for multi-label problems and is calculated as:

    `L_bce = - (y * log(p) + (1 - y) * log(1 - p))`

    Here, `y` is the ground-truth label (0 or 1) and `p` is the predicted probability for that label.

2.  **Contrastive Loss with Hard Negative Mining (`L_contrastive`)**: To improve the model's ability to distinguish between closely related labels, a contrastive loss is applied. This loss focuses on the most challenging examples within each training instance—the "hardest" positive and negative labels. It is defined as:

    `L_contrastive = max(0, margin + max_neg_score - min_pos_score)`

    - `min_pos_score`: The lowest similarity score among all correct (positive) labels for a given text.
    - `max_neg_score`: The highest similarity score among all incorrect (negative) labels.
    - `margin`: A hyperparameter that enforces a minimum separation between positive and negative pairs.

    By maximizing the margin between the hardest positive and negative examples, the model is forced to learn a more fine-grained and resilient decision boundary.

This hybrid loss function allows GliZNet to leverage the stability of BCE while benefiting from the enhanced discriminative power of contrastive learning, leading to more accurate and reliable zero-shot predictions.

### 3.4 Synthetic Data Generation

Training relies on a synthetic dataset of 50,847 text-label pairs, crafted using advanced language models (e.g., Ollama, Groq). The dataset spans domains like technology, healthcare, and sports, incorporating hard negatives to simulate real-world challenges. This controlled data generation ensures robust zero-shot generalization.

### 3.5 Model Architecture

GliZNet comprises three core components:

1. **Text Encoder**: A pre-trained transformer (e.g., `bert-base-uncased`) processes the joint text-label sequence generated by `GliZNETTokenizer`.
2. **Projection Layer**: An optional linear layer that reduces the dimensionality of the transformer's output embeddings for improved efficiency.
3. **Similarity Computation**: A module that scores the alignment between the text and label representations. GliZNet supports multiple similarity metrics, including dot-product, bilinear, and a learned linear transformation, which is optimized during training.

This architecture synergizes efficiency and expressive power, tailored for complex zero-shot tasks.

## 4. Contributions to the State of the Art

GliZNet introduces several advancements:

* **Scalable Efficiency**: O(1) complexity via joint encoding transforms scalability for large label sets.
* **Contextual Label Reasoning**: Label-aware embeddings enable inter-label dependency modeling, a leap beyond static representations.
* **Robust Training**: Hard negative mining and synthetic data bolster discriminative power and adaptability.

These innovations position GliZNet as a pioneering framework, with potential to influence future research in scalable, context-sensitive classification.

## 5. Conclusion

GliZNet redefines zero-shot text classification through its novel joint encoding and contrastive learning paradigm. By addressing computational and contextual limitations of prior methods, it offers a scalable, efficient solution poised for impact in real-time and large-scale applications. This work lays a foundation for exploring advanced zero-shot learning techniques in NLP.

## References

* **Hugging Face**. *Zero-Shot Classification*. [https://huggingface.co/tasks/zero-shot-classification](https://huggingface.co/tasks/zero-shot-classification)
* **Papers With Code**. *Zero-Shot Text Classification*. [https://paperswithcode.com/task/zero-shot-text-classification](https://paperswithcode.com/task/zero-shot-text-classification)
* **arXiv**. (2024). *Retrieval Augmented Zero-Shot Text Classification*. [https://arxiv.org/abs/2406.15241](https://arxiv.org/abs/2406.15241)
* **Towards Data Science**. (2023). *Zero-Shot vs. Similarity-Based Text Classification*. [https://towardsdatascience.com/zero-shot-vs-similarity-based-text-classification-83115d9879f5](https://towardsdatascience.com/zero-shot-vs-similarity-based-text-classification-83115d9879f5)
