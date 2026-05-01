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

## Competitive Gap Analysis

| Metric | Budget | Brute-force | Target |
|--------|--------|-------------|--------|
| Req/s per instance | 450 | 450 | 450 |
| CPU per instance | 0.45 core | 0.45 core | 0.45 core |
| Time per request | ~1ms | ~5.6ms | ~50μs |
| KNN 3M vectors | N/A | ~5600μs | ~50μs |

**Problem:** Brute-force KNN on 3M vectors takes ~5.6ms, which is 5.6x the 1ms budget per request at 900 req/s.

---

## 5 Competitive Optimizations

### 1. HNSW with CGO (100-500x search speedup) — **CRITICAL**

**Library:** `github.com/sunhailin-Leo/hnswlib-to-go` (CGO bindings for nmslib/hnswlib)

| Metric | Brute-force | HNSW (CGO) |
|--------|------------|------------|
| Build time (3M) | N/A (instant) | ~10s |
| Search time | ~5600μs | ~10-50μs |
| Memory | 168MB | ~250MB |
| Accuracy | Exact | ~99% (configurable) |

**Trade-off:** Requires `gcc`/`g++` in build stage. Acceptable because:
- Docker build is one-time
- Final distroless image doesn't need compiler
- 10s build at startup is within healthcheck timeout (30s)

**Implementation:**
```go
index := hnsw.New(14, 16, 200, 42, 3000000, "l2")
// Build: index.AddBatchPoints(vectors, labels, 4)
// Search: labels, dists := index.SearchKNN(query, 5)
```

### 2. JSON Streaming in Handler (~10-20μs savings)

**Current:** `io.ReadAll` + `json.Unmarshal` (2 copies, extra allocation)
**Target:** `json.NewDecoder(r.Body).Decode(&req)` (single pass, zero copy)

```go
// Before
body, _ := io.ReadAll(r.Body)
var req model.FraudScoreRequest
json.Unmarshal(body, &req)

// After
var req model.FraudScoreRequest
json.NewDecoder(r.Body).Decode(&req)
```

### 3. Pre-encoded Response (~15-30μs savings)

**Current:** `json.Marshal(resp)` allocates and encodes every request
**Target:** Manual byte construction — fraud score is always a float, approved is bool

```go
// Before: json.Marshal → ~30μs, 1 alloc
// After:  manual encode → ~2μs, 0 allocs
buf := make([]byte, 0, 64)
buf = strconv.AppendFloat(buf, fraudScore, 'f', 1, 64)
// write: {"approved":true,"fraud_score":0.X}
```

### 4. GC Tuning (reduce p99 spikes)

**Settings:**
- `GOGC=500` — less frequent GC, higher memory tolerance
- `GOMEMLIMIT=140MiB` — hard limit per instance (2 × 140 = 280, leaves 70MB for nginx)
- `GODEBUG=gcstoptheworld=0` — ensure no forced STW

**Impact:** Reduces GC-induced latency spikes that blow p99. At 150MB limit, GC triggers ~every 100MB allocated, which at 900 req/s × 100 bytes = 90KB/s means very infrequent GC.

### 5. nginx Optimization

**Current:** `least_conn` with `keepalive 64`
**Target:** Tuned for low-latency with minimal overhead

```nginx
upstream api_backend {
    server api-1:9999;
    server api-2:9999;
    keepalive 16;              # Reduced (2 instances only)
    keepalive_timeout 10s;     # Match handler timeout
}

# Handler: disable buffering for faster response
proxy_buffering off;
proxy_http_version 1.1;
```

---

## Implementation Order

| # | Optimization | Est. Impact | Effort |
|---|-------------|-------------|--------|
| 1 | HNSW with CGO | 100-500x | High |
| 2 | JSON streaming | 1.2x | Low |
| 3 | Pre-encoded response | 1.5x | Low |
| 4 | GC tuning | p99 stability | Low |
| 5 | nginx optimization | 1.1x | Low |

---

## Expected Results

| Metric | Brute-force | With all optimizations |
|--------|------------|----------------------|
| KNN search | ~5600μs | ~50μs |
| Handler latency | ~5800μs | ~100μs |
| p99 at 900 req/s | > 2000ms (cut) | < 10ms |
| score_p99 | -3000 | > 2000 |
| Memory | 168MB | ~250MB (within 350MB) |

---

## Success Criteria

| Metric | Target | Current |
|--------|--------|---------|
| HNSW build time | < 30s | ~10s (estimated) |
| Search time | < 100μs | ~50μs (estimated) |
| p99 | < 10ms | N/A |
| score_p99 | > 2000 | N/A |
| failure_rate | < 15% | N/A |
| Memory usage | < 300MB | ~168MB |
