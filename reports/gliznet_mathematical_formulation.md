# GliZNet: Mathematical Formulation and Architecture

## Abstract

GliZNet (Generalized Label-Informed Zero-Shot Network) is a novel architecture for zero-shot text classification that leverages label semantics through a carefully designed sequence construction, dual projection spaces, and a multi-objective loss function combining supervised contrastive learning, label repulsion, and binary cross-entropy. This document provides a comprehensive mathematical formulation of the model, detailing how each component contributes to the overall effectiveness.

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

## 4. Dual Projection Architecture

### 4.1 Motivation

Raw hidden states from the backbone live in a space optimized for masked language modeling, not for discriminative similarity computation. GliZNet introduces separate projection spaces for text and labels to:
- Create specialized representations for each modality
- Enable learnable similarity metrics
- Reduce dimensionality if needed (optional)

### 4.2 Text Projection

$$\mathbf{z}_i^{\text{text}} = \mathcal{P}_{\text{text}}(\mathbf{h}_i) = \text{Dropout}(\mathbf{W}_{\text{text}} \mathbf{h}_i + \mathbf{b}_{\text{text}}) \in \mathbb{R}^{d_p}$$

Optional LayerNorm:
$$\mathbf{z}_i^{\text{text}} = \text{LayerNorm}(\mathbf{W}_{\text{text}} \mathbf{h}_i + \mathbf{b}_{\text{text}})$$

where:
- $\mathbf{W}_{\text{text}} \in \mathbb{R}^{d_p \times d_h}$ is the text projection matrix
- $d_p$ is the projected dimension (can equal $d_h$ for identity projection)
- Dropout rate $p \in [0, 1)$ for regularization

### 4.3 Label Projection

$$\mathbf{z}_i^{\text{label}} = \mathcal{P}_{\text{label}}(\mathbf{h}_i) = \text{Dropout}(\mathbf{W}_{\text{label}} \mathbf{h}_i + \mathbf{b}_{\text{label}}) \in \mathbb{R}^{d_p}$$

**Why Separate Projections?**
- Text and labels serve different semantic roles
- Separate weight matrices $\mathbf{W}_{\text{text}} \neq \mathbf{W}_{\text{label}}$ allow the model to learn:
  - Text representation: focus on content, context, and semantic meaning
  - Label representation: focus on category definition, discriminative features
- This dual-space design is inspired by CLIP and other contrastive learning methods

---

## 5. Label Representation Aggregation

### 5.1 Token-Level vs. [LAB]-Token Mode

GliZNet supports two aggregation strategies:

#### **Mode 1: Average Pooling (Default)**

For each label $l_j$, aggregate all its token embeddings:

$$\mathbf{e}_j^{\text{label}} = \frac{1}{|\mathcal{T}_j|} \sum_{i \in \mathcal{T}_j} \mathbf{z}_i^{\text{label}}$$

where $\mathcal{T}_j = \{i \mid m_i = j\}$ is the set of positions belonging to label $j$.

**Mathematical Properties**:
- Permutation-invariant: order of tokens doesn't matter
- Captures full label semantics: multi-word labels benefit from all tokens
- Smooth gradients: all label tokens receive gradients

#### **Mode 2: [LAB] Token Embedding**

Use the special $[\text{LAB}]$ token embedding as the label representation:

$$\mathbf{e}_j^{\text{label}} = \mathbf{z}_{i_j}^{\text{label}} \quad \text{where } i_j \text{ is the position of } [\text{LAB}] \text{ after label } j$$

**Advantages**:
- Faster inference: single token per label
- Simplified computation: no pooling needed
- The $[\text{LAB}]$ token has "seen" all preceding label tokens via self-attention

**Trade-off**: Relies on the transformer's ability to compress label semantics into a single token position.

---

## 6. Text Representation via Token-Level Attention

### 6.1 Motivation

Traditional approaches use the $[\text{CLS}]$ token as a global text representation:

$$\mathbf{e}^{\text{text}} = \mathbf{z}_{\text{CLS}}^{\text{text}}$$

**Limitation**: A single fixed vector may not optimally match different labels. For instance:
- Label "sports" should focus on sport-related words in the text
- Label "politics" should focus on political terms

### 6.2 Label-Specific Text Aggregation

GliZNet computes a **unique text representation for each label** using learned attention:

$$\mathbf{e}_{b,j}^{\text{text}} = \sum_{i \in \mathcal{T}_{\text{text}}^{(b)}} \alpha_{b,j,i} \cdot \mathbf{z}_i^{\text{text}}$$

where:
- $b$ indexes the batch
- $j$ indexes the label
- $\mathcal{T}_{\text{text}}^{(b)}$ is the set of text token positions for sample $b$ (where $m_i = 0$ and attention mask is 1)
- $\alpha_{b,j,i}$ is the attention weight

### 6.3 Attention Weight Computation

The attention scores are computed via dot-product similarity:

$$s_{b,j,i} = \frac{\langle \mathbf{e}_{b,j}^{\text{label}}, \mathbf{z}_i^{\text{text}} \rangle}{\tau_{\text{attn}}}$$

where $\tau_{\text{attn}} > 0$ is a learnable temperature parameter.

**Masking** to exclude non-text positions:

$$\tilde{s}_{b,j,i} = \begin{cases}
s_{b,j,i} & \text{if } i \in \mathcal{T}_{\text{text}}^{(b)} \\
-\infty & \text{otherwise}
\end{cases}$$

**Softmax normalization**:

$$\alpha_{b,j,i} = \frac{\exp(\tilde{s}_{b,j,i})}{\sum_{i' \in \mathcal{T}_{\text{text}}^{(b)}} \exp(\tilde{s}_{b,j,i'})}$$

**Vectorized Implementation**: The code uses batched matrix multiplication for efficiency:

$$\mathbf{E}^{\text{text}} = \text{softmax}\left(\frac{\mathbf{E}^{\text{label}} (\mathbf{Z}^{\text{text}})^T}{\tau_{\text{attn}}}\right) \mathbf{Z}^{\text{text}}$$

where:
- $\mathbf{E}^{\text{label}} \in \mathbb{R}^{N \times d_p}$ contains all label embeddings (N = total labels across batch)
- $\mathbf{Z}^{\text{text}} \in \mathbb{R}^{N \times L \times d_p}$ contains text tokens (broadcasted per label)

**Intuition**: Each label "queries" the text tokens and attends to the most relevant parts, creating a label-conditioned text representation.

---

## 7. Similarity Computation

### 7.1 Similarity Metrics

For each (text, label) pair $(b, j)$, compute a similarity score:

$$\text{sim}_{b,j} = f_{\text{sim}}(\mathbf{e}_{b,j}^{\text{text}}, \mathbf{e}_{b,j}^{\text{label}})$$

GliZNet supports three similarity functions:

#### **7.1.1 Cosine Similarity (Recommended)**

$$\text{sim}_{b,j} = \tau \cdot \frac{\langle \mathbf{e}_{b,j}^{\text{text}}, \mathbf{e}_{b,j}^{\text{label}} \rangle}{\|\mathbf{e}_{b,j}^{\text{text}}\|_2 \cdot \|\mathbf{e}_{b,j}^{\text{label}}\|_2}$$

where:
- $\tau = \exp(\log \tau_0)$ is a learnable temperature (initialized to $\tau_0 = e^{2.0} \approx 7.4$)
- Normalization ensures scores are in $[-\tau, \tau]$, promoting stable gradients

**Why learnable $\tau$?**
- Cosine similarity is bounded in $[-1, 1]$, which may be too conservative
- $\tau$ allows the model to adjust the dynamic range of logits
- Higher $\tau$ → sharper probability distributions (higher confidence)
- Lower $\tau$ → smoother distributions (less confident)

#### **7.1.2 Dot Product**

$$\text{sim}_{b,j} = \mathbf{W}_{\text{dot}} (\mathbf{e}_{b,j}^{\text{text}} \odot \mathbf{e}_{b,j}^{\text{label}}) + b_{\text{dot}}$$

where $\odot$ is element-wise multiplication, and $\mathbf{W}_{\text{dot}} \in \mathbb{R}^{1 \times d_p}$ is a learned weight vector.

#### **7.1.3 Bilinear**

$$\text{sim}_{b,j} = (\mathbf{e}_{b,j}^{\text{text}})^T \mathbf{W}_{\text{bilinear}} \mathbf{e}_{b,j}^{\text{label}} + b_{\text{bilinear}}$$

where $\mathbf{W}_{\text{bilinear}} \in \mathbb{R}^{d_p \times d_p}$ is a learned interaction matrix.

**Trade-offs**:
- **Cosine**: Robust, interpretable, fewer parameters
- **Dot**: Fast, simple, sensitive to magnitude
- **Bilinear**: Most expressive, but highest parameter count ($d_p^2$)

---

## 8. Loss Function: Multi-Objective Optimization

GliZNet's loss is a weighted combination of three complementary objectives:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{SupCon}} \mathcal{L}_{\text{SupCon}} + \lambda_{\text{repulsion}} \mathcal{L}_{\text{repulsion}} + \lambda_{\text{BCE}} \mathcal{L}_{\text{BCE}}$$

where $\lambda_{\text{SupCon}}, \lambda_{\text{repulsion}}, \lambda_{\text{BCE}} \geq 0$ are hyperparameters.

### 8.1 Supervised Contrastive Loss (Primary Objective)

#### **8.1.1 Formulation**

For each sample $b$, let:
- $\mathcal{P}_b = \{j \mid y_{b,j} = 1\}$ be the set of positive (ground truth) labels
- $\mathcal{N}_b$ be all labels (including positives and negatives)

The SupCon loss encourages the model to assign high probability mass to positive labels:

$$\mathcal{L}_{\text{SupCon}} = -\frac{1}{B} \sum_{b=1}^{B} \frac{1}{|\mathcal{P}_b|} \sum_{j \in \mathcal{P}_b} \log \frac{\exp(\text{sim}_{b,j})}{\sum_{k \in \mathcal{N}_b} \exp(\text{sim}_{b,k})}$$

Equivalently, using log-softmax:

$$\mathcal{L}_{\text{SupCon}} = -\frac{1}{B} \sum_{b=1}^{B} \frac{1}{|\mathcal{P}_b|} \sum_{j \in \mathcal{P}_b} \text{log\_softmax}(\text{sim}_{b,:})_j$$

#### **8.1.2 Intuition**

- **Contrastive Nature**: For each positive label, the loss compares its similarity against ALL other labels
- **Ranking**: Encourages positive labels to have higher similarity than negatives
- **Calibration**: Naturally produces calibrated probability distributions via softmax
- **Multi-label Support**: Averaging over all positives handles multiple correct labels

#### **8.1.3 Connection to InfoNCE**

This is a generalization of the InfoNCE loss used in contrastive learning (e.g., SimCLR, CLIP):

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}_{\text{pos}})}{\exp(\text{sim}_{\text{pos}}) + \sum_{k} \exp(\text{sim}_{k}^{\text{neg}})}$$

In GliZNet:
- "Positive" = ground truth labels
- "Negatives" = other labels in the candidate set

**Why SupCon?** It provides:
- Strong discriminative gradients
- Robustness to label imbalance
- Effective representation learning through contrastive structure

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

### 8.3 Binary Cross-Entropy Loss (Auxiliary)

#### **8.3.1 Formulation**

Standard per-label BCE with **decoupled temperature**:

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{b,j} \left[ y_{b,j} \log \sigma(\tilde{s}_{b,j}) + (1 - y_{b,j}) \log(1 - \sigma(\tilde{s}_{b,j})) \right]$$

where:
- $y_{b,j} \in \{0, 1\}$ is the ground truth label
- $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function
- $\tilde{s}_{b,j} = \frac{\text{sim}_{b,j}}{\tau} \cdot \tau_{\text{BCE}}$ is the rescaled logit

**Decoupling**: 
- $\text{sim}_{b,j}$ is scaled by $\tau$ from SupCon
- We unscale by dividing by $\tau$, then apply BCE-specific scale $\tau_{\text{BCE}}$
- This prevents SupCon temperature from dominating BCE gradients

#### **8.3.2 Why Add BCE?**

- **Complementary Signal**: BCE provides direct per-label supervision, while SupCon is comparative
- **Calibration**: BCE encourages well-calibrated probabilities for individual labels
- **Stability**: Helps when SupCon gradients are noisy (e.g., few labels per sample)

#### **8.3.3 Interaction with SupCon**

The two losses have complementary gradient flows:

$$\frac{\partial \mathcal{L}_{\text{SupCon}}}{\partial \text{sim}_{b,j}} = p_{b,j}^{\text{softmax}} - \mathbb{1}[j \in \mathcal{P}_b]$$

$$\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial \text{sim}_{b,j}} = \sigma(\text{sim}_{b,j}) - y_{b,j}$$

- SupCon: Relative gradient (depends on distribution over all labels)
- BCE: Absolute gradient (independent per label)

Together, they provide both **ranking** and **thresholding** signals.

---

## 9. Training Dynamics and Optimization

### 9.1 Gradient Flow

The composite loss creates a rich gradient landscape. For label embedding $\mathbf{e}_j^{\text{label}}$:

$$\frac{\partial \mathcal{L}_{\text{total}}}{\partial \mathbf{e}_j^{\text{label}}} = \lambda_{\text{SupCon}} \frac{\partial \mathcal{L}_{\text{SupCon}}}{\partial \text{sim}_j} \frac{\partial \text{sim}_j}{\partial \mathbf{e}_j^{\text{label}}} + \lambda_{\text{repulsion}} \frac{\partial \mathcal{L}_{\text{repulsion}}}{\partial \mathbf{e}_j^{\text{label}}} + \lambda_{\text{BCE}} \frac{\partial \mathcal{L}_{\text{BCE}}}{\partial \text{sim}_j} \frac{\partial \text{sim}_j}{\partial \mathbf{e}_j^{\text{label}}}$$

**Three forces**:
1. **SupCon**: Pull positives toward text, push negatives away (contrastive)
2. **Repulsion**: Push different labels apart (geometric)
3. **BCE**: Adjust magnitude for calibrated probabilities (scale)

### 9.2 Learnable Parameters

The model learns:
- $\mathbf{W}_{\text{text}}, \mathbf{b}_{\text{text}}$: Text projection (~$d_h \times d_p$ parameters)
- $\mathbf{W}_{\text{label}}, \mathbf{b}_{\text{label}}$: Label projection (~$d_h \times d_p$ parameters)
- $\tau$: SupCon temperature (1 parameter)
- $\tau_{\text{attn}}$: Attention temperature (1 parameter)
- $\tau_{\text{BCE}}$: BCE temperature (1 parameter)
- Similarity head weights (depends on metric)
- Backbone weights (fine-tuned)

**Total new parameters**: ~$2 d_h d_p + d_p^2$ (for bilinear) or ~$2 d_h d_p$ (for cosine/dot)

### 9.3 Optimization Strategy

**Typical hyperparameters**:
- Optimizer: AdamW with weight decay
- Learning rate: 1e-5 to 5e-5 (lower for backbone, higher for new parameters)
- Warmup: 10% of total steps
- Loss weights: $\lambda_{\text{SupCon}} = 1.0$, $\lambda_{\text{BCE}} = 1.0$, $\lambda_{\text{repulsion}} = 0.1$

**Scheduler**: Linear decay after warmup to prevent overfitting

---

## 10. Inference and Prediction

### 10.1 Forward Pass

Given test sample $x$ and candidate labels $\mathcal{L} = \{l_1, \ldots, l_K\}$:

1. **Tokenize**: Construct sequence with [CLS], text, [SEP], labels with [LAB] separators
2. **Encode**: $\mathbf{H} = \mathcal{F}_{\text{backbone}}(\text{seq})$
3. **Project**: Compute $\mathbf{z}_i^{\text{text}}$ and $\mathbf{z}_i^{\text{label}}$ for all positions
4. **Aggregate labels**: Compute $\mathbf{e}_j^{\text{label}}$ for each label $j$
5. **Attention**: Compute label-specific text representations $\mathbf{e}_j^{\text{text}}$ via attention
6. **Similarity**: Compute $\text{sim}_j$ for each label
7. **Probability**: Apply sigmoid: $p_j = \sigma(\text{sim}_j)$

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
2. **Dual projection**: Separates text and label spaces, allowing specialized learned metrics
3. **Label-conditioned attention**: Each label focuses on relevant text parts, avoiding dilution from irrelevant content
4. **Contrastive learning**: SupCon provides strong discriminative signal via relative comparisons
5. **Repulsion**: Prevents collapse while respecting contextual differences
6. **BCE calibration**: Ensures well-calibrated per-label probabilities

**Mathematical Guarantees**:

- **Lipschitz continuity**: With bounded weights and cosine similarity, the model is Lipschitz-continuous in input space
- **Universal approximation**: Bilinear similarity can approximate any scoring function (given sufficient $d_p$)
- **Optimization**: The loss is differentiable everywhere (except at repulsion threshold, but ReLU is subdifferentiable)

### 11.2 Comparison to Related Approaches

| Method | Sequence | Passes | Label Interaction | Loss |
|--------|----------|--------|-------------------|------|
| **Cross-Encoder** | [CLS] text [SEP] label | $K$ | None (independent) | BCE |
| **Dual-Encoder** | [CLS] text; [CLS] label | 2 | None | Contrastive |
| **GliZNet** | [CLS] text [SEP] labels [LAB] ... | 1 | Full (self-attention) | SupCon + Repulsion + BCE |

**GliZNet advantages**:
- Captures label dependencies (e.g., "sports" and "competition" co-occurrence)
- Faster than cross-encoder ($1 \times$ vs. $K \times$ forward passes)
- Richer than dual-encoder (labels interact with text and each other)

---

## 12. Hyperparameter Sensitivity

### 12.1 Loss Weights

**$\lambda_{\text{SupCon}}$**: Primary signal
- Higher → stronger ranking, better discrimination
- Lower → risk of poor calibration

**$\lambda_{\text{BCE}}$**: Calibration
- Higher → better calibrated probabilities, may overfit
- Lower → under-calibrated, but better representations

**$\lambda_{\text{repulsion}}$**: Diversity
- Higher → more separated labels, risk of over-separation
- Lower → risk of collapse

**Recommended**: Start with $(1.0, 1.0, 0.1)$ and tune based on validation.

### 12.2 Temperature Parameters

**$\tau$ (SupCon)**: Controls logit scale
- Higher → sharper distributions (high confidence)
- Lower → smoother distributions (low confidence)
- Initialized to $e^{2.0} \approx 7.4$ (empirically effective)

**$\tau_{\text{attn}}$**: Controls attention sharpness
- Higher → focus on few tokens
- Lower → spread across many tokens
- Initialized to $1.0$

**$\tau_{\text{BCE}}$**: BCE-specific scale
- Decoupled from SupCon to prevent gradient conflicts

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
- **Dual projections**: Separate learned spaces for text and label semantics
- **Label-conditioned attention**: Dynamic text aggregation per label
- **Multi-objective learning**: Balancing discrimination (SupCon), diversity (repulsion), and calibration (BCE)

The mathematical formulation reveals how these components interact:
- Contrastive learning provides strong discriminative gradients
- Repulsion prevents collapse while respecting context
- Attention enables fine-grained text-label matching
- Temperature scaling controls confidence and calibration

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
| $\mathcal{L}_{\text{SupCon}}$ | Supervised contrastive loss | scalar |
| $\mathcal{L}_{\text{repulsion}}$ | Label repulsion loss | scalar |
| $\mathcal{L}_{\text{BCE}}$ | Binary cross-entropy loss | scalar |

---

## Appendix B: Implementation Details

### B.1 Efficient Batched Attention

```python
# Pseudocode for vectorized label-specific attention
def compute_label_attention(text_embeddings, label_embeddings, text_mask):
    """
    text_embeddings: (B, L, D)
    label_embeddings: (N, D) where N = total labels across batch
    text_mask: (N, L) - text positions for each label's batch
    """
    # Scores: (N, 1, D) @ (N, D, L) -> (N, 1, L) -> (N, L)
    scores = torch.bmm(
        label_embeddings.unsqueeze(1),  # (N, 1, D)
        text_embeddings.transpose(1, 2)  # (N, D, L)
    ).squeeze(1) / temperature
    
    # Mask and softmax
    scores = scores.masked_fill(~text_mask, float('-inf'))
    attn_weights = F.softmax(scores, dim=1)  # (N, L)
    
    # Aggregate: (N, 1, L) @ (N, L, D) -> (N, 1, D) -> (N, D)
    text_repr = torch.bmm(
        attn_weights.unsqueeze(1),  # (N, 1, L)
        text_embeddings  # (N, L, D)
    ).squeeze(1)
    
    return text_repr
```

### B.2 Label Aggregation with Scatter

```python
def aggregate_labels(hidden_states, lmask):
    """
    hidden_states: (B, L, D)
    lmask: (B, L) where lmask[i, j] = label_id (0 for non-label)
    """
    label_mask = lmask > 0
    token_label_ids = lmask[label_mask]  # (N_tokens,)
    label_hidden = hidden_states[label_mask]  # (N_tokens, D)
    
    # Aggregate by label_id using scatter
    max_label_id = token_label_ids.max().item()
    aggregated = torch.zeros(max_label_id, D, device=device)
    counts = torch.zeros(max_label_id, device=device)
    
    aggregated.index_add_(0, token_label_ids - 1, label_hidden)
    counts.index_add_(0, token_label_ids - 1, torch.ones(len(token_label_ids)))
    
    return aggregated / counts.unsqueeze(-1)
```

---

**Document Version**: 1.0  
**Last Updated**: December 26, 2025  
**Author**: Generated from GliZNet codebase analysis
