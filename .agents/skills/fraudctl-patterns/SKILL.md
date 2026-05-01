---
name: fraudctl-patterns
description: fraudctl-specific patterns including KNN vector search optimization, 14D vectorization, zero-allocation patterns, and performance best practices for high-throughput fraud detection APIs.
origin: fraudctl
---

# fraudctl Development Patterns

Patterns and best practices specific to the fraudctl fraud detection API.

## When to Activate

- Working on KNN search optimization
- Adding new vector dimensions
- Optimizing hot paths in fraud-score handler
- Benchmarking performance-critical code
- Adding unit tests for vectorizer or KNN

## Architecture Overview

### Data Flow

```
HTTP Request (JSON)
    ↓
[Handler] Parse JSON
    ↓
[Vectorizer] → 14D float32 vector (model.Vector14)
    ↓
[KNN BruteForce] → top-5 neighbors by euclidean distance
    ↓
[Fraud Score] = fraud_neighbors / 5
    ↓
HTTP Response { approved, fraud_score }
```

### Performance Characteristics

No caching is used (prohibited for test-data.json by Rinha 2026 FAQ). All requests go through KNN.

| Path | Latency (100k vectors) | Latency (3M vectors) |
|------|----------------------|---------------------|
| KNN Search (single core) | ~0.19ms | ~5.6ms |
| KNN Search (multi-core) | ~0.15ms | ~3.5ms |
| HTTP Handler (p50) | ~0.3ms | ~6ms |

### Key Components

| Component | File | Responsibility |
|-----------|------|----------------|
| Handler | `internal/handler/fraud_score.go` | HTTP parsing + vectorization + KNN prediction |
| Vectorizer | `internal/vectorizer/vectorizer.go` | JSON → 14D float32 vector |
| KNN | `internal/knn/knn.go` | Optimized brute-force vector search |
| Dataset | `internal/dataset/dataset.go` | In-memory reference vectors + config |
| Loader | `internal/dataset/loader.go` | Streaming JSON loader for references.json.gz |

## KNN Search Pattern

### Zero-Allocation Brute-Force with Early Exit

```go
func (bf *BruteForce) Predict(query model.Vector14, k int) float64 {
    // Stack-allocated arrays - zero heap allocations
    var dists [5]float32
    var frauds [5]bool
    count := 0

    for i := 0; i < n; i++ {
        // Find max distance slot first
        maxIdx := findMaxIndex(dists, count, k)

        // Early exit: compute distance incrementally
        d0 := query[0] - v[0]
        dist := d0 * d0
        if dist >= dists[maxIdx] { continue }
        d1 := query[1] - v[1]
        dist += d1 * d1
        if dist >= dists[maxIdx] { continue }
        // ... continue for all 14 dimensions

        // Replace if closer
        if dist < dists[maxIdx] {
            dists[maxIdx] = dist
            frauds[maxIdx] = bf.labels[i]
        }
    }
    // Count fraud and return ratio
}
```

### Early Exit Optimization

The key optimization is incremental distance computation with early bail-out:

```go
d0 := query[0] - v[0]
dist := d0 * d0
if dist >= maxDist { continue }  // Bail early
d1 := query[1] - v[1]
dist += d1 * d1
if dist >= maxDist { continue }  // Bail early
// ... repeat for all 14 dimensions
```

This reduces average distance computation by ~70% for large datasets.

### Stack Allocation (Zero Heap)

```go
// Stack-allocated - NO heap allocation
var dists [5]float32
var frauds [5]bool

// Avoid: make() which allocates on heap
// bad: dists := make([]float32, 5)
```

## Vectorization Pattern

### 14 Dimensions

| Idx | Dimension | Range | Note |
|-----|-----------|-------|-------|
| 0 | amount (normalized) | [0, 1] | amount / max_amount |
| 1 | installments (normalized) | [0, 1] | installments / max_installments |
| 2 | amount vs avg (normalized) | [0, 1] | ratio / max_ratio |
| 3 | hour of day | [0, 1] | hour / 23 |
| 4 | day of week | [0, 1] | weekday / 6 |
| 5 | minutes since last tx | [-1, 1] | -1 if null |
| 6 | km from last tx | [-1, 1] | -1 if null |
| 7 | km from home | [0, 1] | normalized |
| 8 | tx count 24h | [0, 1] | normalized |
| 9 | is online | {0, 1} | binary |
| 10 | card present | {0, 1} | binary |
| 11 | unknown merchant | {0, 1} | binary |
| 12 | MCC risk score | [0, 1] | lookup table |
| 13 | merchant avg amount | [0, 1] | normalized |

### Pre-computed Inverse Constants

```go
func New(norm model.NormalizationConstants, mccRisk model.MCCRisk) *Vectorizer {
    return &Vectorizer{
        invMaxAmount:   float32(1.0 / norm.MaxAmount),
        invMaxInstall:  float32(1.0 / norm.MaxInstallments),
        // ... pre-compute all inverses to avoid division in hot path
    }
}

// In hot path: multiply instead of divide
vec[0] = clampFloat32(float32(amount) * v.invMaxAmount)
// vs: vec[0] = float32(amount) / v.norm.MaxAmount  // slower
```

### Vector Type

```go
// Fixed-size 14D vector - 56 bytes vs 112 bytes for []float64
type Vector14 [14]float32
```

### clampFloat32 Function

```go
func clampFloat32(val float32) float32 {
    if val < 0 { return 0 }
    if val > 1 { return 1 }
    return val
}
```

## Handler Pattern

### Fallback on Error (No HTTP 500)

```go
func (h *FraudScoreHandler) Handle(w ResponseWriter, r *http.Request) error {
    body, err := io.ReadAll(r.Body)
    if err != nil {
        return h.sendFallback(w)  // Returns 200 with default response
    }
    // ...
}

func (h *FraudScoreHandler) sendFallback(w ResponseWriter) error {
    w.WriteHeader(http.StatusOK)
    _, _ = w.Write([]byte(`{"approved":true,"fraud_score":0.0}`))
    return nil
}
```

### sync.Pool for Response Objects

```go
type FraudScoreHandler struct {
    responsePool sync.Pool
}

func NewFraudScoreHandler(vec Vectorizer, knn KNNPredictor) *FraudScoreHandler {
    h := &FraudScoreHandler{}
    h.responsePool.New = func() interface{} {
        return &model.FraudScoreResponse{}
    }
    return h
}
```

### Small Focused Interfaces

```go
type Vectorizer interface {
    Vectorize(req *model.FraudScoreRequest) model.Vector14
}

type KNNPredictor interface {
    Predict(vector model.Vector14, k int) float64
    Count() int
}
```

## Benchmarking Pattern

### Measure Hot Paths

```bash
# Run benchmarks with profiling
go test -bench=BenchmarkBruteForce_Predict -benchtime=5s \
    -cpuprofile=cpu.prof -memprofile=mem.prof ./internal/knn/

# Analyze with pprof
go tool pprof --text cpu.prof
```

### Key Benchmarks

| Benchmark | Target | Current (100k) | Current (3M est) |
|-----------|--------|----------------|------------------|
| KNN Predict (single) | < 1ms | ~0.19ms | ~5.6ms |
| KNN Predict (parallel) | < 0.5ms | ~0.12ms | ~3.5ms |
| HTTP Handler | < 2ms | ~0.3ms | ~6ms |
| Allocations | 0 | 0 B/op | 0 B/op |

## Performance Quick Reference

| Pattern | Status | Impact |
|---------|--------|--------|
| Early exit distance calc | ✅ Implemented | Very High (70% fewer ops) |
| Unrolled 14D euclidean | ✅ Implemented | High |
| Stack-allocated top-k | ✅ Implemented | High (0 allocs) |
| float32 vectors | ✅ Implemented | Medium (50% memory) |
| Pre-computed inverses | ✅ Implemented | Medium |
| sync.Pool responses | ✅ Implemented | Medium |
| Streaming JSON loader | ✅ Implemented | High (memory for 3M) |
| No HTTP 500 | ✅ Implemented | Critical (weight 5) |

## Testing Pattern

### Table-Driven Tests

```go
func TestBruteForce_Predict(t *testing.T) {
    tests := []struct {
        name    string
        query   model.Vector14
        k       int
        wantMin float64
        wantMax float64
    }{
        {"close to fraud", model.Vector14{0.1, 0.1, ...}, 5, 0.66, 1.0},
        {"close to legit", model.Vector14{0.9, 0.9, ...}, 5, 0.0, 0.34},
    }
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            got := bf.Predict(tt.query, tt.k)
            if got < tt.wantMin || got > tt.wantMax {
                t.Errorf("Predict() = %v, want [%v, %v]", got, tt.wantMin, tt.wantMax)
            }
        })
    }
}
```

### Benchmark Tests

```go
func BenchmarkBruteForce_Predict_100k(b *testing.B) {
    // Setup with 100k vectors
    bf := NewBruteForce()
    bf.Build(vectors, labels)
    query := model.Vector14{0.5, 0.5, ...}

    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _ = bf.Predict(query, 5)
    }
}
```

## Key Files

```
internal/
├── knn/
│   ├── knn.go           # Optimized brute-force KNN
│   └── knn_test.go      # Unit tests + benchmarks
├── vectorizer/
│   ├── vectorizer.go    # 14D float32 vectorization
│   └── vectorizer_test.go
├── handler/
│   ├── fraud_score.go   # HTTP handler
│   └── fraud_score_test.go
├── dataset/
│   ├── dataset.go       # Dataset management
│   └── loader.go        # Streaming JSON loader
└── model/
    └── reference.go     # Vector14 type + models
```

## Quick Reference

| Check | Command |
|-------|---------|
| Run tests | `go test ./... -v` |
| Run benchmarks | `go test -bench=. -benchtime=3s ./...` |
| Check coverage | `go test -cover ./...` |
| Run k6 test | `./scripts/run-k6-test.sh` |
| Profile CPU | `go test -cpuprofile=cpu.prof -bench=.` |
| Profile memory | `go test -memprofile=mem.prof -bench=.` |

**Remember**: No caching allowed. All requests go through KNN. Early exit optimization is critical for performance at 3M vectors.
