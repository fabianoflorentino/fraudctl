---
title: "Go idiomático"
description: "Padrões de engenharia, organização, qualidade e operação que sustentam a performance do fraudctl."
weight: 50
---

## Princípios de engenharia

Performance elevada não exige código de difícil manutenção. O fraudctl concentra as otimizações no caminho crítico e mantém o restante do projeto aderente às convenções da linguagem: pacotes pequenos, responsabilidades claras, erros explícitos e testes com race detector.

## Organização por responsabilidade

| Pacote | Responsabilidade |
|---|---|
| `internal/rawhttp` | Servidor HTTP orientado a bytes |
| `internal/vectorizer` | Conversão de JSON para `Vector14` |
| `internal/knn` | Índice Grid, KD-tree e kernels |
| `internal/handler` | Orquestração da decisão |
| `internal/dataset` | Carregamento de recursos |
| `internal/model` | Tipos compartilhados |
| `cmd/lb` | Balanceador com passagem de descritor |
| `cmd/api` | Composição da aplicação |

## Interfaces no consumidor

O handler depende de duas interfaces reduzidas:

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

Essa separação permite testar o handler com dublês e mantém o pacote `knn` independente da camada HTTP.

## Tratamento de erros

O loader utiliza sentinelas (`ErrInvalidPath`, `ErrReadFile`, `ErrOpenGzip`, `ErrDecodeJSON`) e as envolve com `%w`, permitindo `errors.Is` nos testes. No caminho crítico, falhas de parse não geram erro HTTP: a API responde `200 OK` com o fallback, pois erros HTTP têm peso maior no score do que falsos positivos.

## Qualidade e testes

- testes table-driven nos pacotes principais;
- `go test -race` na integração contínua;
- testes de integração da passagem de file descriptor;
- testes diferenciais do índice contra busca bruta;
- benchmarks separados para `PredictRaw`, vectorizer e dataset.

## Build e operação

- `CGO_ENABLED=0` e binário estático;
- imagem final distroless, com usuário não-root;
- `-trimpath`, `-s -w` e PGO;
- `GOMAXPROCS=1` por container para reduzir overhead de scheduling;
- healthcheck próprio (`-healthcheck`) utilizado pelo compose.

## Exceções justificadas

Assembly AVX2, buffers reutilizados e respostas pré-computadas fogem do Go "de livro". A regra adotada foi **otimização localizada, isolada e medida**, com fallback em Go puro para outras arquiteturas. O restante do código permanece legível, testável e passível de revisão.
