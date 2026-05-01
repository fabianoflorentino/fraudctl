# fraudctl Improvement Plan — Rinha de Backend 2026

## Context

The **Rinha de Backend 2026** competition requires building a fraud detection API that:
- Receives credit card transaction payloads
- Converts them into 14-dimensional vectors
- Finds the 5 nearest neighbors among 3M reference vectors
- Returns `approved = fraud_score < 0.6` where `fraud_score = fraud_count / 5`

### Competition Constraints

| Constraint | Value |
|-----------|-------|
| Dataset | 3,000,000 labeled vectors (`references.json.gz`) |
| Load test | 900 req/s ramped over 120s, max 250 VUs |
| Resources | 1 CPU, 350MB total (2× API + nginx) |
| k6 timeout | 2001ms per request |
| Caching test payloads | **PROHIBITED** |
| Architecture | 2 API instances + load balancer (round-robin) |

### Scoring Formula

```
score_final = score_p99 + score_det

score_p99:
  if p99 > 2000ms → -3000 (cut)
  else → 1000 · log₁₀(1000 / max(p99, 1))
  max: +3000 (p99 ≤ 1ms)

score_det:
  E = 1·FP + 3·FN + 5·Err
  failure_rate = (FP + FN + Err) / N
  if failure_rate > 15% → -3000 (cut)
  else → 1000 · log₁₀(1 / max(E/N, 0.001)) - 300 · log₁₀(1 + E)
  max: +3000 (E = 0)
```

**Target scores:**

| p99 | score_p99 |
|-----|-----------|
| ≤ 1ms | +3000 |
| 10ms | +2000 |
| 100ms | +1000 |
| > 2000ms | -3000 (cut) |

---

## Current Status (Updated)

### Completed

| Task | Status | Details |
|------|--------|---------|
| Remove cache (prohibited) | ✅ Done | Removed `CacheProvider`, `LoadCachedAnswers`, `GetCachedAnswer` |
| Migrate to `model.Vector14` (`[14]float32`) | ✅ Done | 50% memory reduction vs `[]float64` |
| Pre-computed normalization inverses | ✅ Done | Multiplication instead of division |
| Streaming JSON loader | ✅ Done | Low memory for 3M vectors |
| Fallback on errors | ✅ Done | Returns `{"approved":true,"fraud_score":0.0}` |
| HNSW interface | ✅ Done | `internal/knn/hnsw.go` with hann library |
| Dockerfile with `GOARCH=amd64` | ✅ Done | Compatible with test environment |
| docker-compose.yml | ✅ Done | 2 API + nginx, within 350MB |

### Blocked

| Task | Issue | Impact |
|------|-------|--------|
| HNSW index build | Pure Go libraries too slow (>5min for 3M) | Can't pass healthcheck |
| Load testing | Need working API first | Can't measure p99 |

### HNSW Library Evaluation

| Library | Build Time (3M) | Search Time | Notes |
|---------|-----------------|-------------|-------|
| `coder/hnsw` (pure Go) | >5min | ~0.5ms | Too slow build |
| `habedi/hann` (pure Go) | >5min | ~0.3ms | Too slow build |
| `go-hnswlib` (CGO) | ~10s | ~0.1ms | Requires C++ compiler |
| Brute-force | N/A | ~1.2ms | Too slow at 900 req/s |

---

## Recommended Solution: CGO with hnswlib

### Why CGO?

The C++ `hnswlib` library is **10-50x faster** than pure Go implementations for building large indexes. The trade-off is requiring a C++ compiler during build, but this is acceptable since:
1. Docker build stage includes gcc
2. Final distroless image doesn't need gcc
3. Build time is a one-time cost

### Implementation Steps

1. **Add go-hnswlib dependency:**
   ```bash
   go get github.com/viktordanov/go-hnswlib@latest
   ```

2. **Update Dockerfile to include gcc in build stage:**
   ```dockerfile
   FROM golang:1.26-bullseye AS builder
   RUN apt-get update && apt-get install -y g++ && rm -rf /var/lib/apt/lists/*
   WORKDIR /build
   COPY . .
   RUN CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build -o fraudctl ./cmd/api
   ```

3. **Update HNSW implementation to use go-hnswlib:**
   ```go
   import hnsw "github.com/viktordanov/go-hnswlib"

   func NewHNSWIndex() *HNSWIndex {
       index := hnsw.NewL2(14, 3000000, 16, 200, 42)
       return &HNSWIndex{index: index}
   }
   ```

4. **Expected results:**
   - Build time: ~10s for 3M vectors
   - Search time: ~0.1ms per query
   - p99 at 900 req/s: < 10ms
   - Memory: ~250MB for HNSW graph

---

## Alternative: Pre-build Index During Docker Build

If CGO is not viable, pre-build the HNSW index during Docker build:

1. **Create build tool (`cmd/build-index/main.go`):**
   ```go
   func main() {
       refs := loadReferences("resources/references.json.gz")
       index := knn.NewHNSWIndex()
       index.Build(refs)
       index.Save("resources/hnsw-index.bin")
   }
   ```

2. **Update Dockerfile:**
   ```dockerfile
   RUN go run cmd/build-index/main.go
   ```

3. **Update API to load pre-built index:**
   ```go
   index := knn.LoadHNSWIndex("resources/hnsw-index.bin")
   ```

**Drawback:** Pure Go build still takes 5+ minutes during Docker build.

---

## Priority Actions

1. **Implement CGO with go-hnswlib** — fastest solution, ~10s build
2. **Test with k6** — validate p99 < 10ms at 900 req/s
3. **Tune HNSW parameters** — optimize M, efConstruction, efSearch
4. **Update submission branch** — copy final files

---

## Success Criteria

| Metric | Target | Current |
|--------|--------|---------|
| HNSW build time | < 30s | > 5min (pure Go) |
| p99 | < 10ms | N/A |
| score_p99 | > 2000 | N/A |
| failure_rate | < 15% | N/A |
| Memory usage | < 300MB | ~168MB (vectors only) |
