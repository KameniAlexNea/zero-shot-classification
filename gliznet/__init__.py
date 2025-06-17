"""
A zero-shot classification system inspired by the GLiNER paper, aiming to classify text (e.g., a sentence about ancient humans using math for navigation) into positive (e.g., "archaeological_findings", "scientific_discovery", "academic_publication") or negative labels (e.g., "historical_research", "science_fiction_story", "mathematics_education").

You've nearly completed a custom BERT-based tokenizer that formats input as `CLS text SEP lab1 SEP lab2 ... labn`.

Built the model, which will:
1. Use the tokenizer to process input.
2. Extract the CLS token (representing the sentence) and the first token of each label (using a label mask).
3. Compute similarity scores between the CLS token and label embeddings.
4. Calculate the loss using ground truth labels.
The focus now is implementing the model, ensuring clear code, while leveraging the tokenizer for zero-shot classification.

"""
from .config import GliZNetConfig
from .model import GliZNetForSequenceClassification


from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING

CONFIG_MAPPING.register("gliznet", GliZNetConfig)
MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.register(GliZNetConfig, GliZNetForSequenceClassification)
