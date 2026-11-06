---
title: "Performance"
description: "Metas de desempenho, resultados oficiais, otimizações com impacto mensurado e trade-offs."
weight: 40
---

## Objetivos de desempenho

A latência responde por metade do score. A meta era manter o p99 abaixo de 1 ms para saturar a componente de latência, ou aproximar-se desse patamar dentro das restrições de CPU e memória.

O resultado oficial foi **p99 de 1,16 ms** e **2.935 pontos** de latência.

## Referência de benchmark

```
BenchmarkGridIndex_PredictRaw-16    32660    115002 ns/op    0 B/op    0 allocs/op
```

O caminho de predição não realiza alocações. Esse invariante orientou as decisões de implementação subsequentes.

## Otimizações adotadas

### 1. Partição com lookup O(1)

A primeira versão localizava a partição por varredura linear. A versão final indexa um array de 256 posições diretamente pela chave:

```go
sameIdx := idx.partitionsByKey[key] // [256]int32
```

O ganho estimado é a eliminação de cerca de 192 comparações por requisição.

### 2. Aritmética int16 no cálculo de bounds

A conversão de cada dimensão para `float32` durante o cálculo de lower bound tinha custo relevante. A versão final mantém a aritmética em `int16`:

```go
diff := int32(p.bboxMin[d]) - int32(qd)
lb += uint64(diff) * uint64(diff)
```

### 3. Kernel AVX2

O assembly `ScanBlock8AVX2` processa oito vetores por bloco, com dois acumuladores independentes, early exit na dimensão 8 e `PREFETCHT0` para o próximo bloco. Em CPUs sem AVX2, a implementação genérica em Go é utilizada automaticamente.

### 4. Respostas pré-computadas

Como o `fraud_score` assume apenas seis valores, as seis respostas HTTP completas são montadas no `init`. O handler devolve um slice já pronto, sem serialização por requisição.

### 5. Gestão de memória

| Formato | Tamanho do índice |
|---|---|
| `float32` | ~168 MB |
| `int16` | ~84 MB |
| `int16` + rótulos bit-packed | ~84,4 MB |

Com `int16`, o índice cabe no orçamento de 170 MB por instância. `GOGC=off`, `GOMEMLIMIT=140MiB` e `debug.SetMemoryLimit(150 << 20)` completam o controle de memória.

## Pipeline de build

O `grid.bin` é gerado durante o build da imagem Docker, não no startup. O binário é compilado com `-pgo=default.pgo` e `-ldflags="-s -w"`, e a imagem final é distroless.

## Trade-offs

- **IVF aproximado** — melhor desempenho em microbenchmark, porém aproximado. A versão final priorizou exatidão.
- **Prefilter GBDT/RF** — modelos de árvore foram treinados e medidos, mas não entraram no caminho final da submissão.

Os detalhes de implementação e os benchmarks completos estão em [PERFORMANCE.md](https://github.com/fabianoflorentino/fraudctl/blob/main/docs/PERFORMANCE.md).
