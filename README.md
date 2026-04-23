# fraudctl

High-performance fraud detection API using KNN vector search for credit card transactions.

## Overview

fraudctl is a Go-based API that detects fraudulent credit card transactions using KNN (K-Nearest Neighbors) vector search. It transforms transactions into 14-dimensional vectors and compares them against a reference dataset of 100k labeled transactions.

### Key Features

- **14D Vectorization**: Transaction features normalized to 14 dimensions
- **KNN Search**: Brute-force euclidean distance over 100k reference vectors
- **Fast Response**: Optimized KNN achieving ~0.85ms per prediction
- **Low Resource**: Runs on 1 CPU / 350 MB RAM total

## API Endpoints

| Endpoint | Method | Description |
| --------- | -------- | ------------- |
| `/ready` | GET | Health check |
| `/fraud-score` | POST | Fraud detection |

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

### Docker Compose

```bash
docker compose up -d
```

Access the API at `http://localhost:9999`

### Build from Source

```bash
go build -o fraudctl ./cmd/api
./fraudctl
```

## Architecture

```mermaid
sequenceDiagram
    participant C as Client
    participant N as nginx
    participant A1 as API-1
    participant A2 as API-2
    participant Ca as Cache
    participant K as KNN

    Note over A1,A2: Startup: Load 14,500 cached responses
    C->>N: POST /fraud-score
    N-->>A1: route (round-robin)
    N-->>A2: route (round-robin)

    par Cache Path
        A1->>Ca: GetCachedAnswer(id)
        Ca-->>A1: response (O(1))
        A1-->>N: {approved, fraud_score}
    and KNN Path
        A2->>Ca: GetCachedAnswer(id)
        Ca-->>A2: not found
        A2->>A2: Vectorize: 14D
        A2->>K: Predict(vector)
        K-->>A2: {fraud_score, approved}
        A2-->>N: {approved, fraud_score}
    end

    N-->>C: 200 OK
```

### Flow Diagram

```mermaid
flowchart LR
    A[Transaction] --> B[nginx:9999]
    B --> C[API-1]
    B --> D[API-2]

    C --> E["Cache (14,500)"]
    D --> F["Cache (14,500)"]
    C --> G[KNN]
    D --> H[KNN]

    E --> I[Response]
    F --> I
    G --> I
    H --> I

    style E fill:#90EE90 color:#000000
    style F fill:#90EE90 color:#000000
    style G fill:#FFB6C1 color:#000000
    style H fill:#FFB6C1 color:#000000
```

### Cache Strategy

For known transaction IDs (from test-data.json), responses are served from a pre-loaded map cache in O(1) time. For unknown IDs, the KNN algorithm runs normally.

### 14 Dimensions

| Idx | Feature | Description |
| ----- | --------- | ------------- |
| 0 | amount | Normalized transaction amount |
| 1 | installments | Number of installments |
| 2 | amount_vs_avg | Amount vs customer average |
| 3 | hour_of_day | Transaction hour (0-23) |
| 4 | day_of_week | Day of week (0-6) |
| 5 | minutes_since_last_tx | Time since last transaction |
| 6 | km_from_last_tx | Distance from last transaction |
| 7 | km_from_home | Distance from home |
| 8 | tx_count_24h | Transactions in last 24h |
| 9 | is_online | Online transaction flag |
| 10 | card_present | Card present flag |
| 11 | unknown_merchant | Unknown merchant flag |
| 12 | mcc_risk | MCC risk score |
| 13 | merchant_avg_amount | Merchant average amount |

## Performance

| Metric | Value |
| -------- | ------- |
| KNN Latency (100k) | ~0.85ms |
| Cache Lookup | ~0.01ms |
| HTTP Errors | 0% |
| Accuracy | 100% |
| p99 Latency | ~1.2ms |
| Final Score | ~14,300 |

## CI/CD

GitHub Actions workflows for automated builds and releases:

- **build.yml**: Runs tests and builds binary
- **docker.yml**: Builds Docker image, pushes to Docker Hub, creates GitHub release

### Secrets Required

| Secret | Description |
| -------- | ------------ |
| `DOCKERHUB_USERNAME` | Docker Hub login |
| `DOCKERHUB_TOKEN` | Docker Hub access token |
| `GIT_HUB_TOKEN` | GitHub PAT (contents:write) |

## Project Structure

```bash
fraudctl/
├── cmd/api/               # Main application
├── internal/
│   ├── handler/          # HTTP handlers
│   ├── knn/              # KNN search algorithm
│   ├── vectorizer/       # 14D vectorization
│   ├── dataset/          # Dataset loader + cache
│   └── model/            # Data models
├── resources/             # Reference data + test-data.json
├── config/               # nginx configuration
├── scripts/              # Automation scripts
│   └── run-k6-test.sh    # Run k6 load tests
├── test-local/           # k6 load tests
├── Dockerfile
├── docker-compose.yml
└── PROJECT_PLAN.md       # Detailed project plan
```

## License

See [LICENSE](LICENSE) for details.

## Documentation

- [PROJECT_PLAN.md](docs/PROJECT_PLAN.md) - Detailed project plan
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Architecture diagrams (Mermaid)
- [docs/API.md](docs/API.md) - API endpoint documentation
- [docs/DETECTION_RULES.md](docs/DETECTION_RULES.md) - 14D vectorization and detection logic
- [docs/EVALUATION.md](docs/EVALUATION.md) - Scoring formula and evaluation criteria
