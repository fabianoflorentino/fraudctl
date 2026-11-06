---
title: "Performance"
description: "Performance goals, official results, measured optimizations, and trade-offs."
weight: 40
---

## Performance goals

Latency accounts for half of the score. The goal was to keep p99 below 1 ms to saturate the latency component, or get as close as possible within the CPU and memory constraints.

The official result was a **p99 of 1.16 ms** and **2,935 latency points**.

## Benchmark reference

```
BenchmarkGridIndex_PredictRaw-16    32660    115002 ns/op    0 B/op    0 allocs/op
```

The prediction path performs no allocations. This invariant guided the subsequent implementation decisions.

## Adopted optimizations

### 1. O(1) partition lookup

The first version located the partition through a linear scan. The final version indexes a 256-position array directly by key:

```go
sameIdx := idx.partitionsByKey[key] // [256]int32
```

The estimated gain is the elimination of roughly 192 comparisons per request.

### 2. int16 arithmetic in bounds computation

Converting each dimension to `float32` during lower-bound computation had a relevant cost. The final version keeps arithmetic in `int16`:

```go
diff := int32(p.bboxMin[d]) - int32(qd)
lb += uint64(diff) * uint64(diff)
```

### 3. AVX2 kernel

The `ScanBlock8AVX2` assembly processes eight vectors per block, with two independent accumulators, early exit at dimension 8, and `PREFETCHT0` for the next block. On CPUs without AVX2, the generic Go implementation is used automatically.

### 4. Precomputed responses

Because `fraud_score` takes only six values, the six complete HTTP responses are built in `init`. The handler returns a ready slice, with no per-request serialization.

### 5. Memory management

| Format | Index size |
|---|---|
| `float32` | ~168 MB |
| `int16` | ~84 MB |
| `int16` + bit-packed labels | ~84.4 MB |

With `int16`, the index fits within the 170 MB budget per instance. `GOGC=off`, `GOMEMLIMIT=140MiB`, and `debug.SetMemoryLimit(150 << 20)` complete memory control.

## Build pipeline

`grid.bin` is generated during the Docker image build, not at startup. The binary is compiled with `-pgo=default.pgo` and `-ldflags="-s -w"`, and the final image is distroless.

## Trade-offs

- **Approximate IVF** — better microbenchmark performance, but approximate. The final version prioritized exactness.
- **GBDT/RF prefilter** — tree models were trained and measured but did not make it into the submission's final path.

Implementation details and complete benchmarks are available in [PERFORMANCE.md](https://github.com/fabianoflorentino/fraudctl/blob/main/docs/PERFORMANCE.md).
