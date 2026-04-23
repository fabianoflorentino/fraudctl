# Architecture Diagrams

## API Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant N as nginx
    participant H as Handler
    participant V as Vectorizer
    participant K as KNN
    participant Ca as Cache

    C->>N: POST /fraud-score
    N->>H: route to API
    Note over H: Parse JSON & Extract ID
    H->>Ca: GetCachedAnswer(id)
    Ca-->>H: response or not found

    alt ID found in cache
        H->>C: 200 OK {approved, fraud_score}
    else ID not found
        H->>V: Vectorize(transaction)
        V-->>H: 14D Vector
        H->>K: Predict(vector)
        K-->>H: {fraud_score, approved}
        H->>C: 200 OK {approved, fraud_score}
    end
```

### Request Processing Paths

```mermaid
flowchart LR
    A[HTTP Request] --> B[Parse JSON]
    B --> C{ID in<br/>Cache?}

    C -->|Yes| D["Cache<br/>(O(1))"]
    D --> E[Return Response]

    C -->|No| F[Vectorizer]
    F --> G[14D Vector]
    G --> H[KNN Search]
    H --> I[Top-5 Neighbors]
    I --> J[Voting]
    J --> K[Fraud Score]
    K --> E
```

## KNN Algorithm

```mermaid
flowchart LR
    Q[Query<br/>14D] --> F[For each ref]
    F --> D[Euclidean Dist]
    D --> C{dist <<br/>top-K?}
    C -->|Yes| U[Update heap]
    C -->|No| N[Skip]
    U --> E{K = 5?}
    E -->|No| F
    E -->|Yes| V[Count fraud]
    V --> S["fraud / 5"]

    style Q fill:#e1f5fe
    style S fill:#e8f5e8
    style D fill:#fffbe6
```

## Performance Comparison

| Path | Latency | Use Case |
|------|---------|----------|
| Cache Hit | ~0.01ms | Known transaction IDs |
| KNN Search | ~0.85ms | Unknown transaction IDs |
| HTTP Overhead | ~0.15ms | Network + parsing |
