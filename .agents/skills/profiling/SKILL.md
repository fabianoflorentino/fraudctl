---
name: profiling
description: Go performance profiling patterns using pprof, benchmarks, and optimization strategies for identifying and fixing performance bottlenecks.
origin: fraudctl
---

# Go Performance Profiling

Patterns for identifying and fixing performance bottlenecks in Go applications.

## When to Activate

- Investigating slow endpoints
- Optimizing hot paths
- Adding new features that may affect performance
- Pre-submission performance validation

## Profiling Workflow

### 1. Identify the Hot Path

```bash
# Run benchmarks with CPU profile
go test -bench=BenchmarkKNN_Predict \
    -benchtime=5s \
    -cpuprofile=cpu.prof \
    -memprofile=mem.prof \
    ./internal/knn/

# Analyze CPU profile
go tool pprof --text cpu.prof | head -30
```

### 2. Read the Profile

```
      flat  flat%   sum%        cum   cum%
    19.04s 66.25% 66.25%     19.05s 66.28%  euclideanDistanceSquared (inline)
     7.80s 27.14% 93.39%     28.17s 98.02%  (*Dataset).Predict
```

| Column | Meaning |
|--------|---------|
| flat | Time in this function only |
| flat% | Percentage of total time |
| sum% | Cumulative percentage |
| cum | Time in this function + children |
| cum% | Cumulative percentage with children |

### 3. Focus on the Top Consumer

**Rule**: Optimize the function consuming the most time first.

In fraudctl:
- `euclideanDistanceSquared` = 66% (inline, already optimal)
- `Predict` = 98% total

## Benchmarking Pattern

### Basic Benchmark

```go
func BenchmarkPredict(b *testing.B) {
    ds := NewDataset(100000)
    query := make([]float64, 14)

    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        ds.Predict(query, 5)
    }
}
```

### Benchmark with Allocations

```go
func BenchmarkPredict(b *testing.B) {
    ds := NewDataset(100000)
    query := make([]float64, 14)

    b.ReportAllocs()
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        ds.Predict(query, 5)
    }
}
```

### Comparing Implementations

```go
func BenchmarkScalar(b *testing.B) {
    // ... baseline implementation
}

func BenchmarkOptimized(b *testing.B) {
    // ... optimized implementation
}
```

Run with benchstat:
```bash
go test -bench=Benchmark -benchtime=3s ./... > old.txt
# make changes
go test -bench=Benchmark -benchtime=3s ./... > new.txt
benchstat old.txt new.txt
```

## Optimization Quick Reference

### High-Impact Patterns

| Pattern | When to Use | Expected Gain |
|---------|-------------|--------------|
| Inline functions | Hot path, small functions | 10-30% |
| sync.Pool | Repeated allocations | 20-50% |
| Preallocate slices | Known size | 10-30% |
| Avoid interface{} | Hot paths | 5-15% |
| Contiguous memory | Cache-sensitive | 10-20% |

### Low-Impact Patterns

| Pattern | Why Low Impact |
|---------|----------------|
| JSON parsing (jsoniter) | < 1% of hot path |
| SIMD (small vectors) | ~1% for 14D vectors |
| Compiler flags | Go already optimal |

## Memory Profiling

```bash
# Profile with rate = 1 (every allocation)
go test -memprofilerate=1 -memprofile=mem.prof -bench=.

go tool pprof --text mem.prof
```

### Reading Memory Profile

```
      flat  flat%   sum%        cum   cum%
28912kB 27.37% 86.66% 28948kB 27.41%  (*vectorPool).put
```

Focus on functions with high `cum` values — they're allocating or retaining memory.

## Quick Reference

| Task | Command |
|------|---------|
| CPU profile | `go test -cpuprofile=cpu.prof -bench=.` |
| Memory profile | `go test -memprofile=mem.prof -bench=.` |
| Analyze text | `go tool pprof --text profile.prof` |
| Analyze web UI | `go tool pprof -http=:8080 profile.prof` |
| Compare benchmarks | `benchstat old.txt new.txt` |

**Remember**: Profile first, optimize what the profile shows. Don't guess — let the data guide you.