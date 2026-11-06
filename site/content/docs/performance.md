---
title: "Performance"
description: "Benchmarks, quantização int16, kernel AVX2, zero alocação e o que realmente moveu o p99."
weight: 40
---

Latência era metade do score. O objetivo era manter o p99 abaixo de 1 ms para saturar a componente de latência — ou chegar o mais perto possível dentro das restrições de CPU e memória.

O resultado oficial foi **p99 de 1,16 ms** e **2935 pontos** de latência.

## Comece pelo benchmark

```
BenchmarkGridIndex_PredictRaw-16    32660    115002 ns/op    0 B/op    0 allocs/op
```

O caminho de predição não aloca. Esse é o invariante que guiou quase todas as decisões seguintes.

## As otimizações que ficaram

### 1. Partição com lookup O(1)

A primeira versão procurava a partição com uma varredura linear. A versão final indexa um array de 256 posições diretamente pela chave:

```go
sameIdx := idx.partitionsByKey[key] // [256]int32
```

Isso elimina cerca de 192 comparações por request.

### 2. Aritmética int16 no bounds

Converter cada dimensão para `float32` durante o cálculo de lower bound custava caro. A versão final mantém tudo em `int16`:

```go
diff := int32(p.bboxMin[d]) - int32(qd)
lb += uint64(diff) * uint64(diff)
```

### 3. Kernel AVX2

O assembly `ScanBlock8AVX2` processa oito vetores por bloco, com dois acumuladores independentes, early exit na dimensão 8 e `PREFETCHT0` para o próximo bloco. Em CPUs sem AVX2, a implementação genérica em Go é usada automaticamente.

### 4. Respostas pré-computadas

Como `fraud_score` só assume seis valores, as seis respostas HTTP completas são montadas no `init`. O handler devolve um slice já pronto.

### 5. Memória sob controle

| Formato | Tamanho do índice |
|---|---|
| `float32` | ~168 MB |
| `int16` | ~84 MB |
| `int16` + labels bit-packed | ~84,4 MB |

Com `int16`, o índice cabe no orçamento de 170 MB por instância. `GOGC=off`, `GOMEMLIMIT=140MiB` e `debug.SetMemoryLimit(150 << 20)` completam o controle de memória.

## Pipeline de build

O `grid.bin` é gerado no build da imagem Docker, não no startup. O binário é compilado com `-pgo=default.pgo` e `-ldflags="-s -w"`, e a imagem final é distroless.

## O que foi descartado

- **IVF aproximado** — mais rápido no microbenchmark, mas aproximado. A versão final preferiu exatidão.
- **Prefilter GBDT/RF** — modelos de árvore foram treinados e medidos, mas não entraram no caminho final da submissão.

Os detalhes de implementação e os benchmarks completos estão em [PERFORMANCE.md](https://github.com/fabianoflorentino/fraudctl/blob/main/docs/PERFORMANCE.md).
