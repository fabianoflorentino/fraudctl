---
title: "Idiomatic Go"
description: "Engineering standards, organization, quality, and operations behind fraudctl's performance."
weight: 50
---

## Engineering principles

High performance does not require hard-to-maintain code. fraudctl concentrates optimizations on the critical path and keeps the rest of the project aligned with language conventions: small packages, clear responsibilities, explicit errors, and race-detector tests.

## Organization by responsibility

| Package | Responsibility |
|---|---|
| `internal/rawhttp` | Byte-oriented HTTP server |
| `internal/vectorizer` | JSON to `Vector14` conversion |
| `internal/knn` | Grid index, KD-tree, and kernels |
| `internal/handler` | Decision orchestration |
| `internal/dataset` | Resource loading |
| `internal/model` | Shared types |
| `cmd/lb` | Load balancer with descriptor passing |
| `cmd/api` | Application composition |

## Consumer-side interfaces

The handler depends on two small interfaces:

```go
type Vectorizer interface {
    Vectorize(req *model.FraudScoreRequest) model.Vector14
    VectorizeJSON(data []byte) (model.Vector14, error)
}

type KNNIndex interface {
    Predict(query model.Vector14, k int) float64
    PredictRaw(query model.Vector14, nprobe int) int
}
```

This separation allows the handler to be tested with doubles and keeps the `knn` package independent from the HTTP layer.

## Error handling

The loader uses sentinel errors (`ErrInvalidPath`, `ErrReadFile`, `ErrOpenGzip`, `ErrDecodeJSON`) and wraps them with `%w`, enabling `errors.Is` in tests. On the critical path, parse failures do not produce HTTP errors: the API responds `200 OK` with the fallback, because HTTP errors weigh more in the score than false positives.

## Quality and testing

- table-driven tests across the main packages;
- `go test -race` in continuous integration;
- integration tests for file-descriptor passing;
- differential tests of the index against brute-force search;
- separate benchmarks for `PredictRaw`, the vectorizer, and the dataset.

## Build and operations

- `CGO_ENABLED=0` and a static binary;
- distroless final image with a non-root user;
- `-trimpath`, `-s -w`, and PGO;
- `GOMAXPROCS=1` per container to reduce scheduling overhead;
- dedicated healthcheck (`-healthcheck`) used by compose.

## Justified exceptions

AVX2 assembly, reused buffers, and precomputed responses depart from textbook Go. The adopted rule was **localized, isolated, and measured optimization**, with a pure-Go fallback for other architectures. The rest of the code remains readable, testable, and reviewable.
