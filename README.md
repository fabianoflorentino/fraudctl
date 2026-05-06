# fraudctl

High-performance fraud detection API for [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026).

## Overview

fraudctl is a pure-Go API that scores credit card transactions for fraud using an IVF (Inverted File Index) KNN search over 3 million labeled reference vectors. The IVF index is built at `docker build` time and baked into the image — startup loads it in memory at boot with zero disk I/O at request time.

### Key Features

- **IVF KNN v4**: `nlist=4096` clusters, `nprobe=16`, `retryExtra=8` — AoS int16 vectors + bit-packed labels
- **2-stage early exit**: dims 0–7 screened first; skip vector if partial distance ≥ worst-K
- **Zero allocations per query**: `0 B/op, 0 allocs/op` (stack-allocated probe arrays)
- **14D Vectorization**: transaction features normalized to float32[14], quantized to int16 at index time
- **Pure Go**: `CGO_ENABLED=0`, `distroless/base-debian12` image, no C dependencies
- **Resource budget**: 2× API (150MB, 0.45 CPU) + nginx (30MB, 0.10 CPU) = 1 CPU / 330MB total
- **No caching**: prohibited by contest rules; every request goes through KNN

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/ready` | GET | Health check — returns 200 once index is loaded |
| `/fraud-score` | POST | Score a transaction |

### Request

```json
{
  "id": "tx-123",
  "transaction": {
    "amount": 150.00,
    "installments": 3,
    "requested_at": "2024-01-15T10:30:00Z"
  },
  "customer": {
    "avg_amount": 100.00,
    "tx_count_24h": 5,
    "known_merchants": ["MERC-001"]
  },
  "merchant": {
    "id": "MERC-001",
    "mcc": "5411",
    "avg_amount": 50.00
  },
  "terminal": {
    "is_online": false,
    "card_present": true,
    "km_from_home": 10.5
  },
  "last_transaction": {
    "timestamp": "2024-01-15T09:00:00Z",
    "km_from_current": 5.0
  }
}
```

### Response

```json
{
  "approved": true,
  "fraud_score": 0.4
}
```

## Quick Start

```bash
docker compose up -d
```

API available at `http://localhost:9999`.

### Build from Source

```bash
# Build IVF index (writes resources/ivf.bin)
go run ./cmd/build-index -resources ./resources -nlist 4096 -iterations 25

# Run API
PORT=9999 RESOURCES=./resources go run ./cmd/api
```

## Architecture

```
HTTP Request (JSON)
    ↓
[nginx] round-robin to API-1 / API-2 (Unix Domain Sockets)
    ↓
[Handler] decode JSON → model.FraudScoreRequest
    ↓
[Vectorizer] → float32[14]  (pre-computed inverse constants, no division)
    ↓
[IVFIndex.PredictRaw] nprobe=16 clusters → 2-stage scan → topK5 sorted
    ↓
[Boundary retry] if fraud count in [2,3]: probe 8 extra clusters
    ↓
[Voting] fraud neighbors / K=5  →  fraud_score  →  approved = score < 0.6
    ↓
HTTP Response { approved, fraud_score }
```

### IVF Index (format v4)

The index is built once at image build time by `cmd/build-index`:

1. Stream `resources/references.json.gz` (3M vectors)
2. Run parallel k-means (`nlist=4096`, 25 iterations, `GOMAXPROCS` workers, PDE early exit)
3. Write `resources/ivf.bin` (magic `0x49564649`, version 4):
   - AoS layout: `vectors[i*DIM+d]` as `int16` (quantized, scale=10000)
   - Bit-packed labels: `labels[i/8] bit (i%8)`

At startup, `dataset.LoadDefault` reads `ivf.bin` into memory.

### IVF Query Path

For each query:
1. **quantizeQuery**: float32[14] → int16[14]
2. **selectProbes**: sorted-insertion over all 4096 centroids → top-16 closest (stack array, 0 allocs)
3. **scanCluster** × 16: for each cluster (~733 vectors each):
   - Stage 1: compute dims 0–7, skip if `dist ≥ worstK`
   - Stage 2: compute dims 8–13, update `topK5` sorted array
4. **Boundary retry**: if `fraudCount ∈ [2,3]`, probe 8 more clusters (edge case accuracy)
5. Return `fraudCount / 5` as fraud score

### 14 Dimensions

| Idx | Feature | Range |
|---|---|---|
| 0 | amount | [0, 1] |
| 1 | installments | [0, 1] |
| 2 | amount vs avg | [0, 1] |
| 3 | hour of day | [0, 1] |
| 4 | day of week | [0, 1] |
| 5 | minutes since last tx | {-1} ∪ [0, 1] |
| 6 | km from last tx       | {-1} ∪ [0, 1] |
| 7 | km from home | [0, 1] |
| 8 | tx count 24h | [0, 1] |
| 9 | is online | {0, 1} |
| 10 | card present | {0, 1} |
| 11 | unknown merchant | {0, 1} |
| 12 | MCC risk score | [0, 1] |
| 13 | merchant avg amount | [0, 1] |

## Performance

| Metric | Value |
|---|---|
| IVF build time (16 CPUs, nlist=4096, 25 iter) | ~54s |
| IVF build time (2 CPUs, CI runner) | ~10min |
| KNN predict (nlist=512 local, nprobe=16) | ~566µs, 0 allocs |
| KNN predict (nlist=4096 Docker, nprobe=16) | ~70µs (estimated) |
| p99 latency (Docker, 0.45 CPU, k6) | ~91ms |
| HTTP errors | 0% |
| Memory per instance | ~150MB |

### Score History

| Version | Official Score | p99 | Notes |
|---|---|---|---|
| v1.0.45 | 1650 | 112ms | IVF nlist=300, CGo AVX2 |
| v1.0.51 | 3443 | — | IVF nlist=1024, CGo AVX2 |
| v1.0.52 | 3434 | 157ms | IVF nlist=1024, CGo AVX2 (throttled) |
| v1.0.53+ | TBD | ~91ms | IVF v4 nlist=4096, pure Go, 0 allocs |

### Latest Local Docker Result (nlist=4096, nprobe=16)

- p99: `91ms`
- p99 score: `1039`
- detection score: `2910` (FP=1, FN=0, weighted_errors=1)
- **final score: `3949`**

## Project Structure

```
fraudctl/
├── cmd/
│   ├── api/            # HTTP server entrypoint (fasthttp)
│   └── build-index/    # IVF index builder (runs at docker build time)
├── internal/
│   ├── handler/        # HTTP handler (fasthttp, pre-allocated JSON responses)
│   ├── knn/            # IVFIndex (v4 format), BruteAVX2Index, ivf_build, ivf_search
│   ├── vectorizer/     # 14D float32 vectorization (zero-alloc, inverse constants)
│   ├── dataset/        # LoadDefault (ivf.bin loader, SetNProbe/SetRetry config)
│   └── model/          # Vector14, FraudScoreRequest/Response, NormalizationConstants
├── resources/          # references.json.gz (3M vectors); ivf.bin (gitignored, built by Docker)
├── scripts/            # docker-up.sh
├── docs/               # Architecture, API, detection rules, evaluation
├── Dockerfile          # CGO_ENABLED=0, nlist=4096, iterations=25
└── docker-compose.yml  # 2× API + nginx, UDS sockets, GOMAXPROCS=2, GOGC=off
```

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — flow and KNN diagrams
- [docs/API.md](docs/API.md) — full endpoint spec
- [docs/DETECTION_RULES.md](docs/DETECTION_RULES.md) — vectorization and scoring logic
- [docs/EVALUATION.md](docs/EVALUATION.md) — contest scoring formula

## License

See [LICENSE](LICENSE) for details.
