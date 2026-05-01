# fraudctl

High-performance fraud detection API for [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026).

## Overview

fraudctl is a pure-Go API that scores credit card transactions for fraud using an IVF (Inverted File Index) KNN search over 3 million labeled reference vectors. The IVF index is built at `docker build` time and baked into the image — startup loads it in ~148ms with zero disk I/O at request time.

### Key Features

- **IVF KNN**: K=300 clusters, nprobe=1 — ~67μs per prediction over 3M vectors
- **14D Vectorization**: Transaction features normalized to float32[14]
- **Pure Go**: `CGO_ENABLED=0`, `distroless/static:nonroot` image
- **Resource budget**: 2× API (150MB, 0.45 CPU) + nginx (30MB, 0.10 CPU) = 1 CPU / 330MB total
- **No caching**: prohibited by contest rules; every request goes through KNN
- **p99 = 170ms** (k6 official test, 250 VUs)

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

API available at `http://localhost:9999`. The first `/ready` poll may take up to ~15s while the IVF index loads.

### Build from Source

```bash
# Build IVF index (writes resources/ivf.bin, ~164MB)
go run ./cmd/build-index -nlist 300 -iterations 15

# Run API
PORT=9999 RESOURCES=./resources go run ./cmd/api
```

## Architecture

```
HTTP Request (JSON)
    ↓
[nginx] round-robin to API-1 / API-2
    ↓
[Handler] stream-decode JSON → model.FraudScoreRequest
    ↓
[Vectorizer] → float32[14]  (pre-computed inverse constants, no division)
    ↓
[IVFIndex.Predict] → find nearest centroid → scan ~10k cluster vectors
    ↓
[Voting] fraud neighbors / K  →  fraud_score  →  approved = score < 0.5
    ↓
HTTP Response { approved, fraud_score }
```

### IVF Index

The index is built once at image build time by `cmd/build-index`:

1. Stream `resources/references.json.gz` (3M vectors)
2. Run parallel k-means (K=300, 15 iterations, `GOMAXPROCS` workers)
3. Write `resources/ivf.bin` (magic `0x49564649`, version 1, centroids + per-cluster flat float32 + label bytes)

At startup, `dataset.LoadDefault` memory-maps `ivf.bin` in ~148ms.

### 14 Dimensions

| Idx | Feature | Range |
|---|---|---|
| 0 | amount | [0, 1] |
| 1 | installments | [0, 1] |
| 2 | amount vs avg | [0, 1] |
| 3 | hour of day | [0, 1] |
| 4 | day of week | [0, 1] |
| 5 | minutes since last tx | [-1, 1] |
| 6 | km from last tx | [-1, 1] |
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
| IVF build time | ~27s (16 CPUs) |
| Index load time | ~148ms |
| KNN predict (IVF, 3M) | ~67μs |
| HTTP handler p50 | ~50μs |
| p99 latency (250 VUs) | 170ms |
| HTTP errors | 0% |
| F1 score | ~97.5% |
| Memory per instance | ~147MB |

## Project Structure

```
fraudctl/
├── cmd/
│   ├── api/            # HTTP server entrypoint
│   └── build-index/    # IVF index builder (runs at docker build time)
├── internal/
│   ├── handler/        # HTTP handler (stream decode + manual JSON encode)
│   ├── knn/            # IVFIndex + BruteIndex (tests/fallback)
│   ├── vectorizer/     # 14D float32 vectorization
│   ├── dataset/        # LoadDefault (ivf.bin → BruteIndex fallback)
│   └── model/          # Vector14, FraudScoreRequest/Response, NormalizationConstants
├── resources/          # references.json.gz (3M vectors); ivf.bin (gitignored, built by Docker)
├── config/             # nginx.conf
├── scripts/            # docker-up.sh, run-k6-test.sh
├── docs/               # Architecture, API, detection rules, evaluation
├── Dockerfile
└── docker-compose.yml
```

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — flow and KNN diagrams
- [docs/API.md](docs/API.md) — full endpoint spec
- [docs/DETECTION_RULES.md](docs/DETECTION_RULES.md) — vectorization and scoring logic
- [docs/EVALUATION.md](docs/EVALUATION.md) — contest scoring formula

## License

See [LICENSE](LICENSE) for details.
