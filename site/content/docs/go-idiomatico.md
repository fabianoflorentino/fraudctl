---
title: "Go idiomático"
description: "Como o fraudctl mantém performance extrema sem abandonar convenções, testes e legibilidade."
weight: 50
---

Performance extrema não exige código hostil. O fraudctl concentra as otimizações no hot path e mantém o restante do projeto idiomático: pacotes pequenos, responsabilidades claras, erros explícitos e testes com race detector.

## Pacotes por responsabilidade

| Pacote | Responsabilidade |
|---|---|
| `internal/rawhttp` | Servidor HTTP orientado a bytes |
| `internal/vectorizer` | JSON → `Vector14` |
| `internal/knn` | Índice Grid, KD-tree e kernels |
| `internal/handler` | Orquestração da decisão |
| `internal/dataset` | Carregamento de recursos |
| `internal/model` | Tipos compartilhados |
| `cmd/lb` | Load balancer com FD passing |
| `cmd/api` | Composição da aplicação |

## Interfaces no consumidor

O handler depende de duas interfaces pequenas:

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

Isso permite testar o handler com dublês e mantém o pacote `knn` independente de HTTP.

## Erros explícitos

O loader usa sentinelas (`ErrInvalidPath`, `ErrReadFile`, `ErrOpenGzip`, `ErrDecodeJSON`) e envolve com `%w`, permitindo `errors.Is` nos testes. No hot path, falhas de parse não viram erro HTTP: a API responde `200 OK` com o fallback, porque erro HTTP pesa mais no score do que um falso positivo.

## Testes

- testes table-driven em todos os pacotes principais;
- `go test -race` no CI;
- testes de integração da passagem de file descriptor;
- testes diferenciais do índice contra busca bruta;
- benchmarks separados para `PredictRaw`, vectorizer e dataset.

## Build e operação

- `CGO_ENABLED=0`, binário estático;
- imagem final distroless, usuário não-root;
- `-trimpath`, `-s -w` e PGO;
- `GOMAXPROCS=1` por container para evitar overhead de scheduling;
- healthcheck próprio (`-healthcheck`) usado pelo compose.

## O que não é idiomático — e por quê

Assembly AVX2, buffers reutilizados e respostas pré-computadas fogem do Go "de livro". A regra adotada foi: **otimização localizada, isolada e medida**, com fallback em Go puro para outras arquiteturas. O resto do código continua legível, testável e fácil de revisar.
