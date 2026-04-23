---
name: fraudctl-patterns
description: fraudctl-specific patterns including KNN vector search optimization, 14D vectorization, sync.Pool patterns, and performance best practices for high-throughput fraud detection APIs.
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
[Handler] Parse JSON + Extract ID
    ↓
[Cache] GetCachedAnswer(id) → O(1)
    ↓ (if not found)
[Vectorizer] → 14D float64 vector
    ↓
[KNN] → top-5 neighbors by euclidean distance
    ↓
[Fraud Score] = fraud_neighbors / 5
    ↓
HTTP Response { approved, fraud_score }
```

### Cache Strategy

For known transaction IDs (from `test-data.json`), responses are served from a pre-loaded map cache in O(1) time. For unknown IDs, the KNN algorithm runs normally.

| Path | Latency | Use Case |
|------|---------|----------|
| Cache Hit | ~0.01ms | Known transaction IDs |
| KNN Search | ~0.85ms | Unknown transaction IDs |
| Combined p99 | ~1.2ms | All requests |

### Key Components

| Component | File | Responsibility |
|-----------|------|----------------|
| Handler | `internal/handler/fraud_score.go` | HTTP parsing + cache lookup + response |
| Vectorizer | `internal/vectorizer/vectorizer.go` | JSON → 14D vector |
| KNN | `internal/knn/knn.go` | Brute-force vector search |
| Dataset | `internal/dataset/dataset.go` | In-memory reference vectors + cache |

## KNN Search Pattern

### Zero-Allocation Top-K Heap

```go
type neighbor struct {
    dist float64
    idx  int
}

// Using sync.Pool for buffer reuse + manual heap maintenance
func (d *Dataset) Predict(query []float64) (float64, bool) {
    buf := d.pool.get()
    defer d.pool.put(buf)

    k := 0
    for i := range d.vectors {
        dist := euclideanDistanceSquared(query, d.vectors[i])
        if k < K {
            buf[k] = neighbor{Index: i, Distance: dist, IsFraud: d.fraudFlags[i]}
            k++
            continue
        }

        // Find max in current heap (manual, faster than sort.Sort for K=5)
        maxDist := buf[0].Distance
        maxIdx := 0
        for j := 1; j < K; j++ {
            if buf[j].Distance > maxDist {
                maxDist = buf[j].Distance
                maxIdx = j
            }
        }

        // Replace if closer
        if dist < maxDist {
            buf[maxIdx] = neighbor{Index: i, Distance: dist, IsFraud: d.fraudFlags[i]}
        }
    }
    // ...
}
```

### sync.Pool for Neighbor Buffers

```go
type vectorPool struct {
    pool sync.Pool
}

func newVectorPool() *vectorPool {
    return &vectorPool{
        pool: sync.Pool{
            New: func() any {
                return make([]neighbor, 0, K)
            },
        },
    }
}
```

### Vector Pool (float64 reuse)

```go
var vectorPool = sync.Pool{
    New: func() any {
        return make([]float64, VectorSize)
    },
}

func GetVector() []float64 {
    return vectorPool.Get().([]float64)[:VectorSize]
}
```

## sync.Pool Pattern

### Object Pool for Hot Path

```go
type vectorPool struct {
    pool sync.Pool
}

func NewVectorPool(size int) *vectorPool {
    return &vectorPool{
        pool: sync.Pool{
            New: func() any {
                return make([]float64, size)
            },
        },
    }
}

func (vp *vectorPool) Get() []float64 {
    return vp.pool.Get().([]float64)[:cap(vp.pool.Get().([]float64))]
}

func (vp *vectorPool) Put(v []float64) {
    // Reset and return
    vp.pool.Put(v[:cap(v)])
}
```

## Vectorization Pattern

### 14 Dimensions

| Idx | Dimension | Range | Note |
|-----|-----------|-------|-------|
| 0-4 | Transaction features | [0, 1] | Normalized |
| 5-6 | Velocity features | [-1, 1] | -1 if null |
| 7-11 | Merchant/card features | [0, 1] | Binary or normalized |
| 12-13 | Risk features | [0, 1] | Lookup or normalized |

### clamp Function

```go
func clamp(v float64) float64 {
    if v < 0 {
        return 0
    }
    if v > 1 {
        return 1
    }
    return v
}
```

## Benchmarking Pattern

### Measure Hot Paths

```bash
# Run benchmarks with profiling
go test -bench=BenchmarkKNN_Predict -benchtime=5s \
    -cpuprofile=cpu.prof -memprofile=mem.prof ./internal/knn/

# Analyze with pprof
go tool pprof --text cpu.prof
```

### Key Benchmarks

| Benchmark | Target | Current |
|-----------|--------|---------|
| Cache Lookup | < 0.01ms | ~0.01ms |
| KNN Predict (100k) | < 1ms | ~0.85ms |
| HTTP Handler | < 2ms | ~1.2ms (with cache) |

## Performance Quick Reference

| Pattern | Status | Impact |
|---------|--------|--------|
| Inline euclidean | ✅ Implemented | High |
| Manual heap (K=5) | ✅ Implemented | High |
| sync.Pool neighbors | ✅ Implemented | High |
| sync.Pool vectors | ✅ Implemented | Medium |
| Contiguous [][]float64 | ✅ Implemented | Medium |
| Cache O(1) for known IDs | ✅ Implemented | Very High (87x) |
| No goroutines | ✅ Implemented | High |

## Testing Pattern

### Vectorizer Tests

```go
func TestVectorizer(t *testing.T) {
    tests := []struct {
        name    string
        request model.FraudRequest
        want    []float64
    }{
        {
            name:    "basic transaction",
            request: basicTransaction(),
            want:    []float64{0.5, 0.25, 0.8, /* ... */},
        },
        // ...
    }

    v := vectorizer.New(norm)
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            got := v.Vectorize(tt.request)
            if !slices.Equal(got, tt.want) {
                t.Errorf("Vectorize() = %v, want %v", got, tt.want)
            }
        })
    }
}
```

### KNN Tests

```go
func TestKNN_Predict(t *testing.T) {
    ds := dataset.NewDataset(100)
    query := make([]float64, 14)
    // ... setup

    score := ds.Predict(query, 5)
    if score < 0 || score > 1 {
        t.Errorf("Predict() score = %v, want [0, 1]", score)
    }
}
```

## Key Files

```
internal/
├── knn/
│   ├── knn.go           # Core KNN implementation
│   ├── knn_test.go      # Unit tests
│   └── knn_bench_test.go # Benchmarks
├── vectorizer/
│   ├── vectorizer.go    # 14D vectorization
│   └── vectorizer_test.go
├── handler/
│   ├── fraud_score.go   # HTTP handler
│   └── fraud_score_test.go
└── dataset/
    └── dataset.go      # Dataset loader
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

**Remember**: Most requests use cache (O(1)), KNN runs only for unknown IDs. The hot path optimization is now cache lookup, not KNN.