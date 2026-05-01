# Architecture

## Request Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant N as nginx
    participant H as Handler
    participant V as Vectorizer
    participant K as IVFIndex

    C->>N: POST /fraud-score
    N->>H: round-robin to API-1 or API-2
    H->>H: stream-decode JSON
    H->>V: Vectorize(request)
    V-->>H: float32[14]
    H->>K: Predict(vector, k=5)
    K-->>H: fraud_score
    H->>C: 200 OK {approved, fraud_score}
```

## IVF KNN Algorithm

```mermaid
flowchart LR
    Q[Query float32-14] --> C[Find nearest centroid\nEuclidean over K=300]
    C --> CL[Load cluster\n~10k vectors]
    CL --> S[Scan cluster\nmin-heap k=5]
    S --> V[Count fraud neighbors]
    V --> R[fraud_score = fraud / 5]

    style Q fill:#e1f5fe,color:#000000
    style R fill:#e8f5e8,color:#000000
    style C fill:#fffbe6,color:#000000
```

## Index Build (docker build time)

```mermaid
flowchart LR
    G[references.json.gz\n3M vectors] --> KM[k-means\nK=300, 15 iters\nparallel workers]
    KM --> B[ivf.bin\n164MB\nbaked into image]
    B --> L[LoadDefault\nstartup ~148ms]
    L --> I[IVFIndex\nin memory]

    style G fill:#e1f5fe,color:#000000
    style B fill:#fffbe6,color:#000000
    style I fill:#e8f5e8,color:#000000
```

## Resource Allocation

| Component | CPU | Memory |
|---|---|---|
| nginx | 0.10 | 30MB |
| api-1 | 0.45 | 150MB |
| api-2 | 0.45 | 150MB |
| **Total** | **1.00** | **330MB** |

Key env vars per API instance: `GOMAXPROCS=1`, `GOGC=500`, `GOMEMLIMIT=140MiB`

## Performance

| Path | Latency |
|---|---|
| Index load (startup) | ~148ms |
| Find nearest centroid (K=300) | ~5μs |
| Scan cluster (~10k vectors) | ~62μs |
| KNN total | ~67μs |
| HTTP handler p50 | ~50μs |
| p99 (250 VUs, k6) | 170ms |
