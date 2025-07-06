
```mermaid
graph TD
    subgraph "Data Preprocessing"
        A["Raw Input<br/>text, labels"] --> B["GliZNETTokenizer<br/>sequence building"];
        B --> C["Token Processing<br/>truncation & padding"];
        C --> D["Tensor Creation<br/>input_ids, attention_mask, lmask"];
    end

    subgraph "GliZNet Forward Pass"
        D --> E["Backbone Encoding<br/>Transformer layers"];
        E --> F["Hidden States<br/>contextual embeddings"];
        F --> G["Representation Computing<br/>CLS & label projections"];
        G --> H["Similarity Computing<br/>dot/bilinear/dot_learning"];
        H --> I["Logits<br/>similarity scores"];
        I --> J{"Training Mode?"};
        J -->|Yes| K["Loss Computation<br/>BCEWithLogitsLoss"];
        J -->|No| L["GliZNetOutput<br/>logits only"];
        K --> M["GliZNetOutput<br/>loss + logits"];
    end

    subgraph "Tokenizer Details"
        N["Sequence Building<br/>[CLS] + text + [SEP] + lab1 + [;] + lab2 + [;]..."] --> B;
        O["Label Masking<br/>0=text, 1,2,3...=label groups"] --> C;
        P["Length Management<br/>truncation & padding to max_length"] --> C;
    end

    subgraph "Model Components"
        Q["Backbone<br/>BERT/RoBERTa/etc"] --> E;
        R["Projection Layer<br/>Linear/Identity"] --> G;
        S["Similarity Function<br/>configurable metric"] --> H;
        T["Loss Function<br/>scaled BCE"] --> K;
    end

    style A fill:#ffecb3
    style D fill:#e1f5fe
    style M fill:#c8e6c9
    style L fill:#c8e6c9
    style J fill:#fff3e0
    style B fill:#f3e5f5
```
