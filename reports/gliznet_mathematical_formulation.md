# GliZNet: Mathematical Formulation and Architecture

## Abstract

GliZNet (Generalized Label-Informed Zero-Shot Network) is a novel architecture for zero-shot text classification that leverages label semantics through a carefully designed sequence construction, a label aggregator with cross-attention, and a multi-objective loss function combining one-vs-negatives softmax ranking, label repulsion, and focal loss. This document provides a comprehensive mathematical formulation of the model, detailing how each component contributes to the overall effectiveness.

---

## 1. Problem Formulation

### 1.1 Zero-Shot Classification Task

Given:
- A text sample $x$ (e.g., a sentence or document)
- A set of candidate labels $\mathcal{L} = \{l_1, l_2, \ldots, l_K\}$ where each $l_i$ is a text string
- No training examples for the specific labels at test time

Objective: Predict a subset $\mathcal{Y} \subseteq \mathcal{L}$ of labels that apply to $x$.

This is a multi-label classification problem where the model must:
1. Understand the semantic content of both text and labels
2. Compute compatibility scores between text and each label
3. Make independent binary decisions for each label

---

## 2. Input Representation and Tokenization

### 2.1 Sequence Construction

GliZNet employs a unified sequence that embeds both text and label information:

$$\text{seq} = [\text{CLS}] \oplus T_x \oplus [\text{SEP}] \oplus T_{l_1} \oplus [\text{LAB}] \oplus T_{l_2} \oplus [\text{LAB}] \oplus \cdots \oplus T_{l_K} \oplus [\text{LAB}]$$

where:
- $T_x = [t_1^x, t_2^x, \ldots, t_{n_x}^x]$ are tokenized text tokens
- $T_{l_i} = [t_1^{l_i}, t_2^{l_i}, \ldots, t_{n_{l_i}}^{l_i}]$ are tokenized label tokens
- $[\text{CLS}]$, $[\text{SEP}]$, $[\text{LAB}]$ are special tokens
- $\oplus$ denotes concatenation

**Key Innovation**: Unlike cross-encoder approaches that create separate sequences for each (text, label) pair, GliZNet processes all labels simultaneously in a single forward pass, enabling:
- Linear computational complexity in the number of labels: $\mathcal{O}(K)$ instead of $\mathcal{O}(K^2)$
- Rich contextual interactions between all labels within the same encoding

### 2.2 Label Mask (lmask)

A parallel mask sequence $\mathbf{m} \in \mathbb{N}^L$ is constructed where $L$ is the total sequence length:

$$m_i = \begin{cases}
0 & \text{if position } i \text{ is text, special token, or } [\text{LAB}] \\
j & \text{if position } i \text{ belongs to label } l_j, \, j \in \{1, \ldots, K\}
\end{cases}$$

This mask enables:
- Efficient identification of tokens belonging to each label
- Label-specific pooling operations
- Gradient flow control during backpropagation

---

## 3. Backbone Encoding

### 3.1 Transformer Encoder

The input sequence is processed by a pretrained transformer backbone $\mathcal{F}_{\text{backbone}}$:

$$\mathbf{H} = \mathcal{F}_{\text{backbone}}(\text{seq}) \in \mathbb{R}^{L \times d_h}$$

where:
- $\mathbf{H} = [\mathbf{h}_1, \mathbf{h}_2, \ldots, \mathbf{h}_L]$ is the sequence of hidden states
- $d_h$ is the hidden dimension of the backbone (e.g., 768 for BERT-base)
- Each $\mathbf{h}_i \in \mathbb{R}^{d_h}$ is the contextualized representation at position $i$

**Critical Property**: Due to self-attention, each hidden state $\mathbf{h}_i$ contains information about the entire sequence, including:
- Bidirectional text context
- All label semantics
- Cross-modal (text-label) interactions

---

## 4. Representation Space

GliZNet uses backbone hidden states with L2 normalisation but without a learned projection layer. All hidden states are unit-normalised before downstream operations:

$$\mathbf{z}_i = \frac{\mathbf{h}_i}{\|\mathbf{h}_i\|_2} \in \mathbb{R}^{d_h}$$

This normalisation stabilises the cross-attention scores and constrains the input magnitude to the bilinear scorer. Note that subsequent operations (dropout during training, fused projections in `LabelContextAttention`, LayerNorm) alter the exact norm of label representations, so downstream vectors are not guaranteed to remain on the unit hypersphere.

A configurable dropout is applied within the `LabelAggregator` before extracting label token representations as a regulariser during training.

---

## 5. Label Representation Aggregation

### 5.1 [LAB] Token Embedding

The hidden state of the $[\text{LAB}]$ separator token that terminates each label span is used as the label representation:

$$\mathbf{e}_j^{\text{label}} = \mathbf{z}_{i_j} \quad \text{where } i_j \text{ is the position of } [\text{LAB}] \text{ after label } j$$

**Properties**:
- Single token per label — no pooling overhead
- The $[\text{LAB}]$ token has attended to all preceding label tokens via self-attention, compressing their semantics into one vector
- Dropout applied before extraction as regularisation during training

---

## 6. Label-Conditioned Text Representation

### 6.1 Motivation

Traditional approaches use the $[\text{CLS}]$ token as a global text representation:

$$\mathbf{e}^{\text{text}} = \mathbf{z}_{\text{CLS}}^{\text{text}}$$

**Limitation**: A single fixed vector may not optimally match different labels. For instance:
- Label "sports" should focus on sport-related words in the text
- Label "politics" should focus on political terms

GliZNet instead computes a **unique text representation for each label** via cross-attention, where the label embedding serves as the query and text tokens are keys/values.

### 6.2 Cross-Attention Primitive: `TextPool(Q, H, mask)`

All text pooling in GliZNet uses the same operation. Given a set of query vectors $\mathbf{Q} \in \mathbb{R}^{B \times K \times d_h}$ (one per label) and the text hidden states $\mathbf{H} \in \mathbb{R}^{B \times L \times d_h}$:

**Step 1 — Scaled dot-product scores** with learnable temperature:

$$s_{b,j,i} = \tau_{\text{attn}} \cdot \langle \mathbf{q}_{b,j},\; \mathbf{z}_i \rangle$$

where $\tau_{\text{attn}} = \exp(\log \sqrt{d_h})$ is a learnable scalar (initialised to $\sqrt{d_h}$). Text token keys $\mathbf{z}_i$ are L2-normalised (§4). Label queries $\mathbf{q}_{b,j}$ come from $[\text{LAB}]$ token embeddings processed through dropout and optionally through `LabelContextAttention`, so their norm is controlled but not strictly 1.

**Step 2 — Masking** to restrict attention to text positions only:

$$\tilde{s}_{b,j,i} = \begin{cases}
s_{b,j,i} & \text{if } i \in \mathcal{T}_{\text{text}}^{(b)} \\
-\infty & \text{otherwise}
\end{cases}$$

where $\mathcal{T}_{\text{text}}^{(b)}$ is the set of text token positions for sample $b$: positions where $m_i = 0$ (not a label-word token), attention mask is 1 (not padding), and the token is not the $[\text{LAB}]$ separator. Formally:

$$\mathcal{T}_{\text{text}}^{(b)} = \{\,i \mid m_i^{(b)} = 0 \;\wedge\; \text{attn\_mask}_i^{(b)} = 1 \;\wedge\; \text{id}_i^{(b)} \neq \text{id}_{[\text{LAB}]}\,\}$$

**Step 3 — Softmax normalisation**:

$$\alpha_{b,j,i} = \frac{\exp(\tilde{s}_{b,j,i})}{\sum_{i' \in \mathcal{T}_{\text{text}}^{(b)}} \exp(\tilde{s}_{b,j,i'})}$$

**Step 4 — Weighted aggregation**:

$$\text{TextPool}(\mathbf{Q}, \mathbf{H}, \text{mask})_{b,j} = \sum_{i \in \mathcal{T}_{\text{text}}^{(b)}} \alpha_{b,j,i} \cdot \mathbf{z}_i$$

**Vectorized form** (batched matrix multiplication):

$$\text{TextPool}(\mathbf{Q}, \mathbf{H}, \text{mask}) = \text{softmax}\!\left(\tau_{\text{attn}} \cdot \mathbf{Q}\,\mathbf{H}^T \odot \text{mask}\right) \mathbf{H}$$

This primitive is called once (without enrichment) or twice (with enrichment) — see below.

### 6.3 Full Pipeline

The pipeline depends on whether label enrichment is enabled (`enrich_labels=True`, current default):

#### Without enrichment (single pass):

$$\mathbf{e}_{b,j}^{\text{text}} = \text{TextPool}(\mathbf{E}^{\text{label}},\; \mathbf{H},\; \text{mask})_{b,j}$$

Each raw label embedding directly queries the text. Done.

#### With enrichment (two passes — current model):

**Pass 1** — Each label independently pools text to get initial evidence:

$$\mathbf{e}_{b,j}^{\text{text}(1)} = \text{TextPool}(\mathbf{E}^{\text{label}},\; \mathbf{H},\; \text{mask})_{b,j}$$

**Fusion** — Concatenate each label with its text evidence and project to $d_h$:

$$\mathbf{f}_{b,j} = \mathbf{W}_{\text{fuse}} [\mathbf{e}_{b,j}^{\text{label}} \| \mathbf{e}_{b,j}^{\text{text}(1)}] + \mathbf{b}_{\text{fuse}}, \quad \mathbf{W}_{\text{fuse}} \in \mathbb{R}^{d_h \times 2d_h}$$

**Cooperative label attention** — Labels attend to each other's fused context (8-head MHA with residual + LayerNorm):

$$\tilde{\mathbf{E}}^{\text{label}} = \text{LayerNorm}\!\left(\mathbf{E}^{\text{label}} + \text{MHA}(Q{=}\mathbf{E}^{\text{label}},\, K{=}\mathbf{F},\, V{=}\mathbf{F})\right)$$

where $\mathbf{F} = [\mathbf{f}_{b,1}, \ldots, \mathbf{f}_{b,K}]$. A padding mask excludes invalid label positions.

**Pass 2** — The enriched labels re-query text with updated semantics:

$$\mathbf{e}_{b,j}^{\text{text}} = \text{TextPool}(\tilde{\mathbf{E}}^{\text{label}},\; \mathbf{H},\; \text{mask})_{b,j}$$

Only Pass 2's output is used for scoring. The enriched embeddings $\tilde{\mathbf{e}}_{b,j}^{\text{label}}$ are also used as the label representation for the bilinear scorer.

**Why two passes?** After Pass 1, label $i$ knows what text evidence label $j$ found (via the fused context in MHA). This lets labels cooperatively route their attention — e.g., if "sports" already claims athletic terms in Pass 1, "competition" can shift focus to other relevant tokens in Pass 2.

---

## 7. Similarity Computation

### 7.1 Similarity Metrics

For each (text, label) pair $(b, j)$, compute a similarity score:

$$\text{sim}_{b,j} = f_{\text{sim}}(\mathbf{e}_{b,j}^{\text{text}}, \mathbf{e}_{b,j}^{\text{label}})$$

GliZNet supports two similarity functions:

#### **7.1.1 Bilinear (Recommended — Current)**

$$\text{sim}_{b,j} = (\mathbf{e}_{b,j}^{\text{text}})^T \mathbf{W}_{\text{bilinear}} \mathbf{e}_{b,j}^{\text{label}} + b_{\text{bilinear}}$$

where $\mathbf{W}_{\text{bilinear}} \in \mathbb{R}^{d_h \times d_h}$ is a learned interaction matrix and $b_{\text{bilinear}}$ is a scalar bias.

Implemented as `nn.Bilinear(d_h, d_h, 1)`. The bilinear form learns asymmetric directional interactions: which dimensions of the text representation should respond to which dimensions of the label representation.

**Why bilinear?**
- Most expressive: captures cross-space interactions that cosine or dot product cannot
- Learns which dimensions of the text representation should interact with which dimensions of the label representation

#### **7.1.2 Cosine Similarity (Alternative)**

$$\text{sim}_{b,j} = \tau \cdot \frac{\langle \mathbf{e}_{b,j}^{\text{text}}, \mathbf{e}_{b,j}^{\text{label}} \rangle}{\|\mathbf{e}_{b,j}^{\text{text}}\|_2 \cdot \|\mathbf{e}_{b,j}^{\text{label}}\|_2}$$

where $\tau = \exp(\log \tau_0)$ is a learnable temperature (initialized to $1/0.07 \approx 14.3$, clamped to 100). The explicit L2-normalisation inside `CosineScoring` normalises the inputs onto the unit sphere regardless of upstream transformations.

**Trade-offs**:
- **Bilinear** *(current, recommended)*: Most expressive; learns a cross-space interaction matrix $\mathbf{W} \in \mathbb{R}^{d_h \times d_h}$; ~$d_h^2$ additional parameters
- **Cosine**: Robust, interpretable, fewer parameters (only 1 learnable scalar); constrains scores to $[-\tau, \tau]$

---

## 8. Loss Function: Multi-Objective Optimization

GliZNet's loss is a weighted combination of three complementary objectives:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{softmax}} \mathcal{L}_{\text{softmax}} + \lambda_{\text{repulsion}} \mathcal{L}_{\text{repulsion}} + \lambda_{\text{focal}} \mathcal{L}_{\text{focal}}$$

where $\lambda_{\text{softmax}} = 1.0$, $\lambda_{\text{repulsion}} = 0.1$, $\lambda_{\text{focal}} = 0.8$ are hyperparameters.

### 8.1 One-vs-Negatives Softmax Loss (Primary Objective)

#### **8.1.1 Formulation**

For each sample $b$, let:
- $\mathcal{P}_b = \{j \mid y_{b,j} = 1\}$ be the set of positive (ground truth) labels
- $\mathcal{N}_b = \{j \mid y_{b,j} = 0,\, \text{valid}\}$ be the valid negative labels

For each positive $p \in \mathcal{P}_b$, the loss is the cross-entropy of that positive against all negatives, with an optional additive margin $m \geq 0$ on the negatives:

$$\ell_{b,p} = \log\!\left(\exp(\text{sim}_{b,p}) + \sum_{n \in \mathcal{N}_b} \exp(\text{sim}_{b,n} + m)\right) - \text{sim}_{b,p}$$

Positives **do not appear** in each other's denominator. The full loss averages over positives per sample, then over samples:

$$\mathcal{L}_{\text{softmax}} = \frac{1}{|\mathcal{B}^+|} \sum_{b \in \mathcal{B}^+} \frac{1}{|\mathcal{P}_b|} \sum_{p \in \mathcal{P}_b} \ell_{b,p}$$

where $\mathcal{B}^+ = \{b \mid |\mathcal{P}_b| > 0\}$.

#### **8.1.2 Intuition**

- **No positive competition**: Under a global softmax, positive labels compete with each other because probabilities must sum to 1. Here each positive is evaluated solely against the negatives.
- **Additive margin**: $+m$ on negative logits forces the model to maintain a gap of at least $m$ before the loss saturates, preventing lazy boundaries when the negative pool is small.
- **Numerical stability**: Implemented as $\text{logsumexp}([\text{sim}_{b,p},\; \text{neg\_lse}]) - \text{sim}_{b,p}$, where $\text{neg\_lse} = \log \sum_{n} \exp(\text{sim}_{b,n} + m)$.

> **Note**: This is *not* Supervised Contrastive Loss (SupCon/InfoNCE). It operates on per-sample classification logits, not on embedding views, and has no contrastive pairs or anchor structure.

### 8.2 Label Repulsion Loss

#### **8.2.1 Motivation**

Without constraints, the label projection network might collapse all label embeddings to a single point:

$$\mathbf{e}_1^{\text{label}} \approx \mathbf{e}_2^{\text{label}} \approx \cdots \approx \mathbf{e}_K^{\text{label}}$$

This "representation collapse" destroys discriminative power. However, we must respect that label embeddings are **contextual** (they depend on the text), so global repulsion across different samples would be incorrect.

#### **8.2.2 Same-Sample Repulsion**

The loss penalizes high similarity between **different labels within the same sample**:

$$\mathcal{L}_{\text{repulsion}} = \frac{1}{B} \sum_{b=1}^{B} \frac{1}{|\mathcal{D}_b|} \sum_{(j, k) \in \mathcal{D}_b} \max\left(0, \text{cosim}(\mathbf{e}_{b,j}^{\text{label}}, \mathbf{e}_{b,k}^{\text{label}}) - \theta_{\text{rep}}\right)$$

where:
- $\mathcal{D}_b = \{(j, k) \mid j \neq k, j, k \in \text{labels of sample } b\}$ are distinct label pairs in sample $b$
- $\text{cosim}(\mathbf{u}, \mathbf{v}) = \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$ is cosine similarity
- $\theta_{\text{rep}} \in [0, 1]$ is a threshold (e.g., 0.3)

**Key Property**: Only penalizes pairs with similarity **above** $\theta_{\text{rep}}$:
- If $\text{cosim} \leq \theta_{\text{rep}}$: no penalty (labels are sufficiently distinct)
- If $\text{cosim} > \theta_{\text{rep}}$: linear penalty encourages further separation

#### **8.2.3 Why Same-Sample Only?**

Consider two samples:
- Sample A: "The team won the championship" with labels ["sports", "competition"]
- Sample B: "The election was competitive" with labels ["politics", "competition"]

The label "competition" should have:
- Different embeddings in A (sports context) vs. B (politics context)
- But separation from "sports" in A and from "politics" in B

Repulsion within sample preserves this contextual sensitivity.

### 8.3 Focal Loss (Auxiliary)

#### **8.3.1 Formulation**

Let $p_{b,j} = \sigma(\text{sim}_{b,j})$ be the predicted probability for sample $b$, label $j$, and let:

$$p_{b,j}^{(t)} = p_{b,j} \cdot y_{b,j} + (1 - p_{b,j})(1 - y_{b,j})$$

be the probability assigned to the ground-truth class. The per-element focal loss is:

$$\ell_{b,j} = (1 - p_{b,j}^{(t)})^{\gamma_b} \cdot \text{BCE}(p_{b,j},\, y_{b,j})$$

where $\text{BCE}(p, y) = -[y \log p + (1-y)\log(1-p)]$, and padding positions ($y_{b,j} = -100$) are excluded.

#### **8.3.2 Adaptive Gamma**

For *pure-class* samples — where $\mathcal{P}_b = \varnothing$ (all-negative) or $\mathcal{N}_b = \varnothing$ (all-positive) — $\mathcal{L}_{\text{softmax}} = 0$ by design, since it requires both classes for its one-vs-negatives formulation. FocalLoss then becomes the **only** source of gradient for those samples. Applying full focal down-weighting ($\gamma = 1.85$) would suppress exactly the signal the model needs. The adaptive gamma disables focusing for pure-class samples:

$$\gamma_b = \begin{cases} \gamma & \text{if } |\mathcal{P}_b| > 0 \text{ and } |\mathcal{N}_b| > 0 \quad \text{(mixed-class)} \\ 0 & \text{otherwise} \quad \text{(standard BCE)} \end{cases}$$

Mixed-class samples retain full $\gamma = 1.85$ for hard-example mining. Pure-class samples revert to standard BCE for reliable, unattenuated gradients.

#### **8.3.3 Class-Balanced Per-Sample Aggregation**

Standard focal loss averages over all valid label positions in a sample:

$$\frac{1}{|\mathcal{P}_b| + |\mathcal{N}_b|} \sum_{j \in \mathcal{P}_b \cup \mathcal{N}_b} \ell_{b,j}$$

In a *needle* scenario (1 positive, 10 negatives), this gives the single positive $\frac{1}{11} \approx 9\%$ of the gradient weight, even though learning what a positive looks like is conceptually as important as learning what a negative looks like.

GliZNet instead computes the mean separately for each class, then combines them with equal weight:

$$\bar{\ell}_b^{+} = \frac{1}{\max(|\mathcal{P}_b|, 1)} \sum_{j \in \mathcal{P}_b} \ell_{b,j}, \qquad \bar{\ell}_b^{-} = \frac{1}{\max(|\mathcal{N}_b|, 1)} \sum_{j \in \mathcal{N}_b} \ell_{b,j}$$

$$c_b = \mathbb{1}[|\mathcal{P}_b| > 0] + \mathbb{1}[|\mathcal{N}_b| > 0], \qquad \mathcal{L}_b^{\text{focal}} = \frac{\bar{\ell}_b^{+} + \bar{\ell}_b^{-}}{c_b}$$

Here $c_b \in \{1, 2\}$: pure-class samples contribute only their one present class; mixed-class samples average the two class means. The batch loss then averages over all samples with at least one valid label ($\mathcal{B}_{\text{valid}} = \{b \mid \exists j : y_{b,j} \neq -100\}$):

$$\mathcal{L}_{\text{focal}} = \frac{1}{|\mathcal{B}_{\text{valid}}|} \sum_{b \in \mathcal{B}_{\text{valid}}} \mathcal{L}_b^{\text{focal}}$$

#### **8.3.4 Interaction with Softmax Loss**

The two losses have complementary gradient flows:

$$\frac{\partial \mathcal{L}_{\text{softmax}}}{\partial \text{sim}_{b,j}} = p_{b,j}^{\text{softmax}} - \mathbb{1}[j \in \mathcal{P}_b]$$

$$\frac{\partial \mathcal{L}_{\text{focal}}}{\partial \text{sim}_{b,j}} = -(1 - p_{b,j}^{(t)})^{\gamma_b}\left[1 + \gamma_b \log p_{b,j}^{(t)}\right] \cdot p_{b,j}^{(t)}(1 - p_{b,j}^{(t)}) \cdot (y_{b,j} - p_{b,j}) \cdot \frac{1}{c_b}$$

- **Softmax loss**: Relative gradient — depends on the full distribution over labels in the sample (ranking signal)
- **Focal loss**: Absolute gradient — amplified for hard examples ($p^{(t)}$ small), zeroed for pure-class samples ($\gamma_b = 0$), class-balanced so minority labels are not diluted

Together, they provide both **discriminative ranking** (which label is most compatible) and **calibrated thresholding** (is each label's probability above a meaningful decision boundary) signals.

---

## 9. Training Dynamics and Optimization

### 9.1 Gradient Flow

The composite loss creates a rich gradient landscape. For label embedding $\mathbf{e}_j^{\text{label}}$:

$$\frac{\partial \mathcal{L}_{\text{total}}}{\partial \mathbf{e}_j^{\text{label}}} = \lambda_{\text{softmax}} \frac{\partial \mathcal{L}_{\text{softmax}}}{\partial \text{sim}_j} \frac{\partial \text{sim}_j}{\partial \mathbf{e}_j^{\text{label}}} + \lambda_{\text{repulsion}} \frac{\partial \mathcal{L}_{\text{repulsion}}}{\partial \mathbf{e}_j^{\text{label}}} + \lambda_{\text{focal}} \frac{\partial \mathcal{L}_{\text{focal}}}{\partial \text{sim}_j} \frac{\partial \text{sim}_j}{\partial \mathbf{e}_j^{\text{label}}}$$

**Three forces**:
1. **One-vs-negatives softmax**: For each positive, push its logit above all negative logits by at least margin $m$ (relative ranking)
2. **Repulsion**: Push different labels apart within the same sample (geometric)
3. **Focal loss**: Class-balanced, scenario-adaptive calibration signal. Full focusing ($\gamma = 1.85$) for mixed-class samples; standard BCE ($\gamma = 0$) for pure-class samples where softmax loss provides no gradient. Positive and negative class losses are averaged separately with equal weight, preventing minority-class dilution.

### 9.2 Learnable Parameters

The model learns:
- $\mathbf{W}_{\text{bilinear}} \in \mathbb{R}^{d_h \times d_h}$, $\mathbf{b}_{\text{bilinear}}$: Bilinear scoring head (~$d_h^2$ parameters)
- Backbone weights (fine-tuned via AdamW)

**Total new parameters (beyond backbone)**: $d_h^2 + d_h$ (bilinear head only)

### 9.3 Optimization Strategy

**Typical hyperparameters**:
- Optimizer: AdamW with weight decay
- Learning rate: 1e-5 to 5e-5 (lower for backbone, higher for new parameters)
- Warmup: 10% of total steps
- Loss weights: $\lambda_{\text{softmax}} = 1.0$, $\lambda_{\text{focal}} = 0.8$ ($\gamma = 1.85$, adaptive), $\lambda_{\text{repulsion}} = 0.1$; margin $m = 0.1$

**Scheduler**: Cosine decay after warmup to prevent overfitting

---

## 10. Inference and Prediction

### 10.1 Forward Pass

Given test sample $x$ and candidate labels $\mathcal{L} = \{l_1, \ldots, l_K\}$:

1. **Tokenize**: Construct sequence with [CLS], text, [SEP], labels with [LAB] separators
2. **Encode**: $\mathbf{H} = \mathcal{F}_{\text{backbone}}(\text{seq})$
3. **Extract labels**: $\mathbf{e}_j^{\text{label}} = \mathbf{h}_{i_j}$ from each `[LAB]` token position
4. **Cross-attend**: Compute label-specific text representations $\mathbf{e}_j^{\text{text}}$ by attending over text tokens with $\mathbf{e}_j^{\text{label}}$ as query
5. **Score**: $\text{sim}_j = \text{Bilinear}(\mathbf{e}_j^{\text{text}}, \mathbf{e}_j^{\text{label}})$
6. **Probability**: Apply sigmoid: $p_j = \sigma(\text{sim}_j)$

### 10.2 Decision Rule

**Binary decision for each label**:

$$\hat{y}_j = \begin{cases}
1 & \text{if } p_j \geq \delta \\
0 & \text{otherwise}
\end{cases}$$

where $\delta \in [0, 1]$ is a threshold (typically 0.5, but can be tuned for precision/recall).

**Top-k selection** (alternative):

$$\hat{\mathcal{Y}} = \text{top}_k(\{p_1, p_2, \ldots, p_K\})$$

Select the $k$ labels with highest probabilities.

### 10.3 Computational Complexity

For batch size $B$, sequence length $L$, and $K$ labels:

- **Tokenization**: $\mathcal{O}(B \cdot L)$
- **Backbone encoding**: $\mathcal{O}(B \cdot L^2 \cdot d_h)$ (due to self-attention)
- **Projection**: $\mathcal{O}(B \cdot L \cdot d_h \cdot d_p)$
- **Label aggregation**: $\mathcal{O}(B \cdot K \cdot d_p)$
- **Attention**: $\mathcal{O}(B \cdot K \cdot L \cdot d_p)$ (batched matrix multiplication)
- **Similarity**: $\mathcal{O}(B \cdot K \cdot d_p)$

**Total**: $\mathcal{O}(B \cdot L^2 \cdot d_h + B \cdot K \cdot L \cdot d_p)$

**Comparison to cross-encoder**: 
- Cross-encoder requires $K$ forward passes: $\mathcal{O}(K \cdot B \cdot L^2 \cdot d_h)$
- GliZNet: Single forward pass with label-specific attention
- **Speedup**: ~$K \times$ faster for large $K$

---

## 11. Theoretical Analysis

### 11.1 Why Does This Work?

**Information Flow**:

1. **Backbone encoding**: Self-attention allows each token to "see" all labels, creating rich contextual embeddings
2. **L2 normalisation**: Places all representations on the unit hypersphere, stabilising attention and scoring
3. **Label-conditioned attention**: Each label focuses on relevant text parts, avoiding dilution from irrelevant content
4. **Label enrichment**: Labels cooperatively route by attending to each other's text evidence
5. **Repulsion**: Prevents collapse while respecting contextual differences
6. **Focal loss**: Focuses on hard examples, improving discrimination on difficult labels

**Mathematical Guarantees**:

- **Lipschitz continuity**: With L2-normalised inputs and bounded bilinear weights, the model is Lipschitz-continuous in input space
- **Universal approximation**: Bilinear similarity can approximate any scoring function (given sufficient $d_h$)
- **Optimization**: The loss is differentiable everywhere (except at repulsion threshold, but ReLU is subdifferentiable)

### 11.2 Comparison to Related Approaches

| Method | Sequence | Passes | Label Interaction | Loss |
|--------|----------|--------|-------------------|------|
| **Cross-Encoder** | [CLS] text [SEP] label | $K$ | None (independent) | BCE |
| **Dual-Encoder** | [CLS] text; [CLS] label | 2 | None | Contrastive |
| **GliZNet** | [CLS] text [SEP] labels [LAB] ... | 1 | Full (self-attention) | Softmax + Repulsion + Focal |

**GliZNet advantages**:
- Captures label dependencies (e.g., "sports" and "competition" co-occurrence)
- Faster than cross-encoder ($1 \times$ vs. $K \times$ forward passes)
- Richer than dual-encoder (labels interact with text and each other)

---

## 12. Hyperparameter Sensitivity

### 12.1 Loss Weights

**$\lambda_{\text{softmax}}$**: Primary signal
- Higher → stronger ranking, better discrimination
- Lower → risk of poor calibration

**$\lambda_{\text{focal}}$**: Hard-example focus
- Higher → stronger focus on hard examples, may overfit
- Lower → less emphasis on hard examples, smoother training

**$\lambda_{\text{repulsion}}$**: Diversity
- Higher → more separated labels, risk of over-separation
- Lower → risk of collapse

**Recommended**: Start with $(1.0, 0.8, 0.1)$ for (softmax, focal, repulsion) and tune based on validation.

### 12.2 Temperature Parameters

**$\tau$ (cosine scoring)**: Controls logit scale (only used if cosine scoring is selected)
- Higher → sharper distributions (high confidence)
- Lower → smoother distributions (low confidence)
- Initialized to $1/0.07 \approx 14.3$

**$\tau_{\text{attn}}$**: Controls attention sharpness
- Higher → focus on few tokens
- Lower → spread across many tokens
- Initialized to $1.0$

**$\tau_{\text{focal}}$**: Focal-specific scale
- Decoupled from softmax to prevent gradient conflicts

### 12.3 Projection Dimension $d_p$

- **$d_p = d_h$**: No dimensionality reduction, maximum capacity
- **$d_p < d_h$**: Regularization via bottleneck, faster
- **$d_p > d_h$**: Possible but rarely beneficial

**Typical choice**: $d_p = d_h$ with dropout for regularization

### 12.4 Repulsion Threshold $\theta_{\text{rep}}$

- **$\theta_{\text{rep}} = 0$**: Penalize all positive cosine similarity
- **$\theta_{\text{rep}} \in [0.3, 0.5]$**: Allow some similarity (recommended)
- **$\theta_{\text{rep}} = 1$**: No repulsion (disable)

**Intuition**: Related labels (e.g., "cat" and "animal") should have some positive similarity.

---

## 13. Extensions and Future Directions

### 13.1 Hierarchical Labels

Extend repulsion loss to respect label hierarchy:

$$\mathcal{L}_{\text{hier}} = \sum_{(j, k) \in \mathcal{H}} \max(0, \theta_{\text{parent}} - \text{cosim}(\mathbf{e}_j, \mathbf{e}_k))$$

where $\mathcal{H}$ contains parent-child pairs, encouraging child labels to be similar to parents.

### 13.2 Few-Shot Adaptation

Use label prototypes from few examples:

$$\mathbf{e}_j^{\text{proto}} = \frac{1}{N_j} \sum_{i=1}^{N_j} \mathbf{e}_{j,i}^{\text{label}}$$

Adapt the model by fine-tuning on these prototypes.

### 13.3 Cross-Lingual Zero-Shot

Replace backbone with multilingual model (e.g., mBERT, XLM-R) to enable:
- Training on English labels
- Inference on text in any language

---

## 14. Conclusion

GliZNet represents a novel synthesis of:
- **Unified encoding**: Efficient single-pass processing of text and all labels
- **L2-normalised representation space**: Stable gradients without learned projections
- **Label-conditioned attention**: Dynamic text aggregation per label
- **Label enrichment**: Cooperative label context via multi-head self-attention
- **Multi-objective learning**: Balancing discrimination (softmax), diversity (repulsion), and hard-example focus (focal loss)

The mathematical formulation reveals how these components interact:
- Contrastive learning provides strong discriminative gradients
- Repulsion prevents collapse while respecting context
- Attention enables fine-grained text-label matching
- Focal loss focuses training on hard examples
- Label enrichment enables cooperative routing between labels

Together, these design choices create a powerful, efficient, and interpretable zero-shot classification architecture.

---

## Appendix A: Notation Summary

| Symbol | Description | Dimension |
|--------|-------------|-----------|
| $x$ | Input text | - |
| $\mathcal{L}$ | Set of candidate labels | - |
| $K$ | Number of labels | scalar |
| $B$ | Batch size | scalar |
| $L$ | Sequence length | scalar |
| $d_h$ | Hidden dimension of backbone | scalar |
| $d_p$ | Projected dimension | scalar |
| $\mathbf{H}$ | Hidden states from backbone | $\mathbb{R}^{L \times d_h}$ |
| $\mathbf{h}_i$ | Hidden state at position $i$ | $\mathbb{R}^{d_h}$ |
| $\mathbf{z}_i^{\text{text}}$ | Projected text representation | $\mathbb{R}^{d_p}$ |
| $\mathbf{z}_i^{\text{label}}$ | Projected label representation | $\mathbb{R}^{d_p}$ |
| $\mathbf{e}_j^{\text{label}}$ | Aggregated label $j$ embedding | $\mathbb{R}^{d_p}$ |
| $\mathbf{e}_j^{\text{text}}$ | Label-conditioned text embedding | $\mathbb{R}^{d_p}$ |
| $\text{sim}_{b,j}$ | Similarity score for sample $b$, label $j$ | scalar |
| $\tau$ | Temperature scale (learnable) | scalar |
| $\mathcal{L}_{\text{softmax}}$ | One-vs-negatives softmax loss | scalar |
| $\mathcal{L}_{\text{repulsion}}$ | Label repulsion loss | scalar |
| $\mathcal{L}_{\text{focal}}$ | Focal loss | scalar |

---

---

**Document Version**: 2.1  
**Last Updated**: May 13, 2026  
**Author**: Alex Kameni
