# Architecture

## Request Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant N as nginx
    participant H as Handler
    participant V as Vectorizer
    participant K as IVFIndex

    C->>N: POST /fraud-score (port 9999)
    N->>H: round-robin via Unix Domain Socket
    H->>H: decode JSON (fasthttp)
    H->>V: Vectorize(request)
    V-->>H: float32[14]
    H->>K: PredictRaw(vector, nprobe=16)
    K->>K: quantizeQuery → int16[14]
    K->>K: selectProbes → top-16 centroids (0 allocs)
    K->>K: scanCluster ×16 (2-stage early exit)
    K->>K: boundary retry if fraudCount ∈ [2,3]
    K-->>H: fraudCount (0..5)
    H->>C: 200 OK {approved, fraud_score}

    alt decode error
        H->>C: 200 OK {approved:true, fraud_score:0.0}
    end
```

## IVF KNN Algorithm

```mermaid
flowchart LR
    Q[Query float32-14] --> QQ[quantizeQuery\nfloat32 → int16\nscale=10000]
    QQ --> SP[selectProbes\nsorted-insertion\nover 4096 centroids\ntop-16 closest\n0 allocs stack array]
    SP --> SC[scanCluster ×16\n~733 vectors each\nStage1: dims 0-7\nearly exit if dist≥worst\nStage2: dims 8-13]
    SC --> TK[topK5\nsorted insertion\nK=5 neighbors]
    TK --> BR{fraudCount\n∈ 2,3 ?}
    BR -->|yes| RE[retry: probe\n8 more clusters]
    RE --> TK
    BR -->|no| FS[fraud_score = fraudCount / 5\napproved = score < 0.6]

    style Q fill:#e1f5fe,color:#000000
    style FS fill:#e8f5e8,color:#000000
    style SP fill:#fffbe6,color:#000000
    style BR fill:#fce4ec,color:#000000
```

## Index Build (docker build time)

```mermaid
flowchart LR
    G[references.json.gz\n3M vectors] --> KM[k-means\nnlist=4096\n25 iterations\nparallel GOMAXPROCS workers\nPDE early exit per centroid]
    KM --> WR[write ivf.bin v4\nAoS int16 vectors\nbit-packed labels\n~84MB]
    WR --> L[LoadDefault\nstartup read into memory]
    L --> I[IVFIndex\nnlist=4096\nnprobe=16\nretryExtra=8]

    style G fill:#e1f5fe,color:#000000
    style WR fill:#fffbe6,color:#000000
    style I fill:#e8f5e8,color:#000000
```

## ivf.bin Format (v4)

```
magic     uint32   0x49564649
version   uint32   4
nlist     uint32   4096
dim       uint32   14
n         uint32   3000000
centroids [nlist×DIM]float32   — cluster centroids
offsets   [nlist+1]uint32      — cluster boundaries
vectors   [n×DIM]int16         — AoS: vectors[i*DIM+d]
labels    [ceil(n/8)]byte      — bit-packed: bit i%8 of byte i/8
```

## Resource Allocation

| Component | CPU | Memory | Env |
|---|---|---|---|
| nginx | 0.10 | 30MB | — |
| api-1 | 0.45 | 150MB | `GOMAXPROCS=2`, `GOGC=off`, `GOMEMLIMIT=120MiB` |
| api-2 | 0.45 | 150MB | `GOMAXPROCS=2`, `GOGC=off`, `GOMEMLIMIT=120MiB` |
| **Total** | **1.00** | **330MB** | budget: 350MB |

Communication between nginx and API instances uses **Unix Domain Sockets** on a shared `tmpfs` volume — eliminates TCP loopback overhead (~40–60µs/request).

## Performance

| Path | Latency |
|---|---|
| selectProbes (4096 centroids) | ~10µs |
| scanCluster ×16 (~733 vectors each, nlist=4096) | ~60µs |
| KNN total (nlist=4096) | ~70µs (estimated) |
| KNN total (nlist=512 local benchmark) | ~566µs |
| p99 (Docker 0.45 CPU, k6, nlist=4096) | ~91ms |
| Allocations per query | **0 B/op, 0 allocs/op** |

## Score History

| Version | Official Score | Local Docker | p99 | Notes |
|---|---|---|---|---|
| v1.0.45 | 1650 | — | 112ms | nlist=300, CGo |
| v1.0.51 | 3443 | — | — | nlist=1024, CGo AVX2 |
| v1.0.52 | 3434 | — | 157ms | nlist=1024, CGo AVX2 (CPU throttled) |
| v1.0.53+ | TBD | **3949** | 91ms | IVF v4, nlist=4096, pure Go, 0 allocs |
