# GliZNet: Zero-Shot Text Classification

GliZNet is a generalist and lightweight model for zero-shot sequence classification, inspired by [GLiNER](https://github.com/urchade/GLiNER) and [GLiClass](https://github.com/Knowledgator/GLiClass). It achieves efficient classification by encoding both text and labels in a single forward pass.

## ✨ Features

- **Zero-Shot Classification**: Works out-of-the-box with pretrained transformers (no task-specific training required)
- **Efficient Architecture**: Single forward pass for all labels (10x faster than cross-encoders)
- **Flexible Design**: Supports multi-label and multi-class classification
- **Multiple Similarity Metrics**: Cosine, dot product, or bilinear similarity
- **Configurable**: Optional projection layers, multiple loss functions for training
- **Production Ready**: Clean pipeline interface inspired by GLiClass

## 🚀 Quick Start

### Installation

```bash
pip install -e .
```

### Basic Usage

```python
from gliznet import GliZNetForSequenceClassification, ZeroShotClassificationPipeline
from gliznet.tokenizer import GliZNETTokenizer

# Load model and tokenizer
model_name = "alexneakameni/gliznet-ModernBERT-base"
model = GliZNetForSequenceClassification.from_pretrained(model_name)
tokenizer = GliZNETTokenizer.from_pretrained(model_name)

# Create pipeline
pipeline = ZeroShotClassificationPipeline(
    model, tokenizer, 
    classification_type='multi-label',
    device='cuda:0'
)

# Classify text
text = "One day I will see the world!"
labels = ["travel", "dreams", "sport", "science", "politics"]
results = pipeline(text, labels, threshold=0.5)

for result in results[0]:
    print(f"{result['label']} => {result['score']:.3f}")
```

### Zero-Shot with Pretrained Backbone

GliZNet works immediately with any pretrained transformer:

```python
from gliznet import GliZNetConfig, GliZNetForSequenceClassification
from gliznet.tokenizer import GliZNETTokenizer

# Default config: no projection, cosine similarity, mean pooling
config = GliZNetConfig(backbone_model="microsoft/deberta-v3-small")
tokenizer = GliZNETTokenizer.from_backbone_pretrained(config.backbone_model)
model = GliZNetForSequenceClassification.from_backbone_pretrained(config, tokenizer)

# Use immediately for zero-shot!
pipeline = ZeroShotClassificationPipeline(model, tokenizer)
results = pipeline("I love this movie!", ["positive", "negative", "neutral"])
```

## 📊 Architecture

- **Backbone**: Any HuggingFace transformer (DeBERTa, RoBERTa, BERT, etc.)
- **Text Encoding**: [CLS] token representation
- **Label Encoding**: Mean pooling over label tokens
- **Similarity**: Configurable (cosine, dot product, bilinear)
- **Projection**: Optional (default: identity, uses raw embeddings)

### Input Format

```
[CLS] text tokens [SEP] label1 [LAB] label2 [LAB] label3 [SEP]
```

## 🎯 Use Cases

- **Sentiment Analysis**: Classify text as positive/negative/neutral
- **Topic Classification**: Categorize documents into topics
- **Intent Detection**: Identify user intent in conversations
- **Content Moderation**: Flag inappropriate content
- **News Categorization**: Organize articles by category
- **Document Organization**: Tag and organize large document collections

## 🛠️ Training

Train on your own data:

```bash
# Single GPU training
python train_gliznet.py \
    --config configs/your_config.yaml \
    --output_dir models/your-model

# Multi-GPU training
bash train_gliznet.sh
```

### Training Data Format

```json
[
  {
    "text": "Sample text here",
    "all_labels": ["label1", "label2", "label3"],
    "true_labels": ["label1", "label3"]
  }
]
```

See `train_gliznet.py` for detailed training configuration.

## 📁 Repository Structure

```
gliznet/                  # Main model package
├── model.py             # GliZNet model implementation
├── tokenizer.py         # Custom tokenizer with label masking
├── predictor.py         # Pipeline interface
├── config.py            # Configuration classes
└── data.py              # Data loading utilities


train_gliznet.py        # Training script
train_gliznet.sh        # Multi-GPU training script
```

## 🔧 Configuration

```python
from gliznet import GliZNetConfig

config = GliZNetConfig(
    backbone_model="microsoft/deberta-v3-small",
    projected_dim=None,              # None = no projection
    similarity_metric="cosine",      # "cosine", "dot", "bilinear"
    use_projection_layernorm=False,  # LayerNorm after projection
    
    # Training loss weights
    bce_loss_weight=1.0,
    supcon_loss_weight=1.0,
    label_repulsion_weight=0.1,
    
    # Temperature scaling
    logit_scale_init=2.0,
    learn_temperature=True,
)
```

## 🤝 Citation

If you use GliZNet in your research, please cite:

```bibtex
@software{gliznet2025,
  title = {GliZNet: Generalized Ligthweights Zero-Shot Text Classification},
  author = {Alex Kameni},
  year = {2025},
  url = {https://github.com/KameniAlexNea/zero-shot-classification}
}
```

## 📝 License

MIT License

## 🙏 Acknowledgments

- Inspired by [GLiNER](https://github.com/urchade/GLiNER) and [GLiClass](https://github.com/Knowledgator/GLiClass)
- Built on [HuggingFace Transformers](https://github.com/huggingface/transformers)
