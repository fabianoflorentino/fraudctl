# fraudctl

High-performance fraud detection API for [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026).

## Overview

fraudctl is a pure-Go API that scores credit card transactions for fraud using an IVF (Inverted File Index) KNN search over 3 million labeled reference vectors. The IVF index is built at `docker build` time and baked into the image — startup loads it in memory at boot with zero disk I/O at request time.

### Key Features

- **IVF KNN v5**: `nlist=4096` clusters, `nprobe=36`, `quickProbe=16` early exit, bbox pruning — AoS int16 vectors + bit-packed labels + cluster bounding boxes
- **2-tier early exit**: quick probe (16 clusters) → re-score only if fraud ∈ {2,3}; ~80-90% of queries exit early
- **Zero-allocation JSON parser**: manual one-pass JSON parse + vectorize (96 B/op vs 697 B/op with gojson)
- **14D Vectorization**: transaction features normalized to float32[14], quantized to int16 at index time
- **Pure Go**: `CGO_ENABLED=0`, `distroless/base-debian12` image, no C dependencies
- **Resource budget**: 2× API (150MB, 0.40 CPU) + haproxy (50MB, 0.20 CPU) = 1 CPU / 350MB total
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
[haproxy] round-robin to API-1 / API-2 (Unix Domain Sockets)
    ↓
[Handler] decode JSON → model.FraudScoreRequest
    ↓
[Vectorizer] → float32[14]  (pre-computed inverse constants, no division)
    ↓
[IVFIndex.PredictRaw] 2-pass adaptive:
  Pass 1: fastNProbe=12 clusters → scan
  Pass 2: only if fraud ∈ {2,3} → scan remaining up to fullNProbe=48
    ↓
[Cluster scan] 2-stage + bbox pruning:
  Stage 1: dims 5,6,2,0,7,1,3,4 → early exit if dist ≥ worst
  Stage 2: dims 8,11,12,9,10,13 → complete distance
    ↓
[Voting] fraud neighbors / K=5  →  fraud_score  →  approved = score < 0.6
    ↓
HTTP Response { approved, fraud_score }
```

### IVF Index (format v5)

The index is built once at image build time by `cmd/build-index`:

1. Stream `resources/references.json.gz` (3M vectors)
2. Run parallel k-means (`nlist=4096`, 32 iterations, `GOMAXPROCS` workers, PDE early exit)
3. Write `resources/ivf.bin` (magic `0x49564649`, version 5):
   - AoS layout: `vectors[i*DIM+d]` as `int16` (quantized, scale=10000)
   - Bit-packed labels: `labels[i/8] bit (i%8)`
   - **Bounding boxes**: `bbox_min[ci*DIM+d]`, `bbox_max[ci*DIM+d]` — per-cluster min/max for bbox pruning
   - SoA centroids: transposed for cache-friendly `selectProbes`

At startup, `dataset.LoadDefault` reads `ivf.bin` into memory and configures:
- `SetNProbe(48)` — full probe count
- `SetRetry(16, 2, 3)` — retry extra if fraud ∈ [2,3] boundary zone

### IVF Query Path (2-pass adaptive + bbox pruning)

For each query:
1. **quantizeQuery**: float32[14] → int16[14]
2. **selectProbes**: sorted-insertion over all 4096 centroids → top-48 closest (stack array, 0 allocs)
3. **2-pass adaptive**:
   - **Pass 1**: scan `fastNProbe=12` clusters (~625 vectors each)
   - If fraud ∉ {2,3} boundary zone → done
   - **Pass 2 (boundary only)**: scan remaining clusters up to `fullNProbe=48`
4. **scanCluster** with 2-stage vector evaluation + bbox pruning:
   - **BBox check**: before scanning, check if cluster bbox could improve top-K
   - **Stage 1 (high-variance dims first)**: 5,6,2,0,7,1,3,4 → early exit if `dist ≥ worstK`
   - **Stage 2 (remaining)**: 8,11,12,9,10,13 → complete distance
   - Block size: 16 vectors per iteration
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
| v1.0.53–v1.0.83 | ~3900–5636 | ~91ms–2.31ms | IVF v4/v5 nlist=4096, pure Go |
| **v1.0.84+** | TBD | TBD | **IVF v5 nlist=4096, nprobe=48, bbox pruning, page-warming REMOVED, pure Go** |

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
│   ├── knn/            # IVFIndex (v5 format with bbox), BruteAVX2Index, ivf_build, ivf_search
│   ├── vectorizer/     # 14D float32 vectorization (zero-alloc, inverse constants)
│   ├── dataset/        # LoadDefault (ivf.bin loader, SetNProbe/SetRetry config)
│   └── model/          # Vector14, FraudScoreRequest/Response, NormalizationConstants
├── resources/          # references.json.gz (3M vectors); ivf.bin (gitignored, built by Docker)
├── scripts/            # docker-up.sh
├── docs/               # Architecture, API, detection rules, evaluation
├── Dockerfile          # CGO_ENABLED=0, GOAMD64=v3, nlist=4096, iterations=32
├── docker-compose.yml  # 2× API + haproxy, UDS sockets, GOMAXPROCS=1, GOGC=off, GOMEMLIMIT=145MiB
└── config/
    └── haproxy.cfg     # Load balancer config (UDS, health checks)
```

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — flow and KNN diagrams
- [docs/API.md](docs/API.md) — full endpoint spec
- [docs/DETECTION_RULES.md](docs/DETECTION_RULES.md) — vectorization and scoring logic
- [docs/EVALUATION.md](docs/EVALUATION.md) — contest scoring formula

## License

See [LICENSE](LICENSE) for details.
