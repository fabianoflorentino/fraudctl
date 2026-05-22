# Performance e Otimizações

Este documento detalha todas as otimizações de performance implementadas no fraudctl, com benchmarks e justificativas.

## Métricas Alvo

| Métrica | Alvo | Notas |
|---------|------|-------|
| p99 local | < 1.10ms | Hardware competição: Mac Mini 2014 (~1.78× multiplier) |
| p99 competição | < 2.00ms | Limite para score negativo (-3000) |
| Alocações/request | 0 | Zero GC pressure no hot path |
| CPU por instância | 0.48 cores | Limite do docker-compose |
| Memória por instância | 170 MB | Limite do docker-compose |

## Hierarquia de Índices

```
grid.bin (KD-tree exato) → ivf.bin (IVF aproximado) → brute force (fallback)
```

O `dataset.LoadDefault` tenta carregar na ordem:
1. `grid.bin` — KD-tree exato, mais rápido
2. `ivf.bin` — IVF aproximado, fallback
3. `references.json.gz` — Brute force, último recurso

---

## Otimizações do Grid Index

### 1. Lookup de Partição O(1)

**Antes:** Busca linear no array de partições (~192 elementos).
```go
// ANTES: O(n) linear scan
var sameIdx int32 = -1
for i := range idx.partitions {
    if idx.partitions[i].key == key {
        sameIdx = int32(i)
        break
    }
}
```

**Depois:** Array direto indexado pela chave.
```go
// DEPOIS: O(1) direct lookup
sameIdx := idx.partitionsByKey[key]  // [256]int32
```

**Impacto:** Elimina ~192 comparações por request.

### 2. Aritmética int16 no Bounds Computation

**Antes:** Conversão int16 → float32 no hot path.
```go
// ANTES: conversão f32 por dimensão
qf := float32(qi[d])
if qf < p.bboxMinF32[d] {
    diff := p.bboxMinF32[d] - qf
    lb += uint64(diff * diff)
}
```

**Depois:** Aritmética int16 direta.
```go
// DEPOIS: aritmética int16
qd := qi[d]
if qd < p.bboxMin[d] {
    diff := int32(p.bboxMin[d]) - int32(qd)
    lb += uint64(diff) * uint64(diff)
}
```

**Impacto:** Elimina 14 conversões f32 por partição escaneada.

### 3. gridBBoxLowerBound Desenrolado

**Antes:** Loop com overhead.
```go
// ANTES: loop
for i := 0; i < 14; i++ {
    qi := int32(q[i])
    if qi < int32(bmin[i]) { ... }
}
```

**Depois:** Código desenrolado para 14 dimensões.
```go
// DEPOIS: unrolled
qi0 := int32(q[0])
if qi0 < int32(bmin[0]) { ... }
qi1 := int32(q[1])
if qi1 < int32(bmin[1]) { ... }
// ... até qi13
```

**Impacto:** O compilador otimiza melhor (sem loop counter, melhor register allocation).

### 4. Early Exit no Bounds Computation

```go
for d := 0; d < 14; d++ {
    // ... calcula diff²
    if lb >= worst { break }  // Early exit
}
```

**Impacto:** Se a distância acumulada já excede o pior candidato, para de calcular.

### 5. Early Exit quando worstDist = 0

```go
if sameIdx >= 0 {
    gridSearchKDTree(p.root, lb, &qi, &h, idx)
    if h.worstDist() == 0 {
        return h.fraudCount(idx.labels)  // Match exato encontrado
    }
}
```

**Impacto:** Se encontrou um vetor idêntico à query (distância 0), não precisa buscar mais.

### 6. SIMD AVX2 com Acumuladores Paralelos

```asm
; Dois acumuladores independentes
VXORPS Y0, Y0, Y0     ; even-dim accumulator
VXORPS Y4, Y4, Y4     ; odd-dim accumulator

; Dim 0 → Y0, Dim 1 → Y4, Dim 2 → Y0, Dim 3 → Y4, ...
; No checkpoint (dim 8): merge Y4 → Y0 e compara com worst
```

**Impacto:**
- 8 vetores processados simultaneamente
- Early exit na dim 8 se todos excedem worst
- Acumuladores paralelos reduzem dependency chain pela metade

### 7. PREFETCHT0 para Próximo Bloco

```asm
PREFETCHT0 256(BX)
PREFETCHT0 320(BX)
PREFETCHT0 384(BX)
```

**Impacto:** Traz próximo bloco de vetores para L1 enquanto FMAs estão em flight.

---

## Otimizações do IVF Index

### 1. Centroids em SoA (Structure of Arrays)

```go
// SoA: centroids[d*nlist + ci]
// Cache-friendly para selectProbes (varre uma dimensão de todos os centroids)
```

### 2. Norma dos Centroids Pré-computada

```go
// ||q - c||² = ||q||² + ||c||² - 2·q·c
// ||c||² pré-computado em LoadIVF, ||q||² calculado por query
```

### 3. Busca 2-Tier com Zona Ambígua

```go
// Pass 1: quickProbe clusters
// Se fraudCount ∈ [boundaryLo, boundaryHi] → Pass 2: nprobe clusters
```

### 4. Bounding Box Pruning por Cluster

```go
if bboxMayImprove(bboxMin, bboxMax, ci, q, worstDist) {
    scanCluster(idx, ci, q, &h)
}
```

### 5. Ordem de Avaliação de Dimensões

```go
// Stage 1: dims de alta variância primeiro (early exit mais provável)
// 5, 6, 2, 0, 7, 1, 3, 4
// Stage 2: dims restantes
// 8, 11, 12, 9, 10, 13
```

---

## Otimizações do rawhttp

### 1. Buffer Pool

```go
var bufPool = sync.Pool{
    New: func() any {
        b := make([]byte, 1024)
        return &b
    },
}
```

### 2. Roteamento por Bytes

```go
// Sem string allocation
case len(path) == 12 && path[1] == 'f':  // /fraud-score
case len(path) == 6 && path[1] == 'r':   // /ready
```

### 3. Respostas Pré-computadas

```go
var fraudResponses [6][]byte  // HTTP completo pré-montado

func init() {
    for i, body := range bodies {
        fraudResponses[i] = buildOKResponse("application/json", body)
    }
}
```

### 4. Parse Manual de SCM_RIGHTS

```go
// Evita alloc de ParseSocketControlMessage
func parseUnixRights(oob []byte) (int, error) {
    if len(oob) < 20 { return -1, nil }
    level := int(oob[8]) | int(oob[9])<<8 | ...
    typ := int(oob[12]) | int(oob[13])<<8 | ...
    if level != syscall.SOL_SOCKET || typ != syscall.SCM_RIGHTS {
        return -1, nil
    }
    fd := int(oob[16]) | int(oob[17])<<8 | ...
    return fd, nil
}
```

---

## Otimizações do Vectorizer

### 1. Multiplicação por Inverso

```go
// ANTES: divisão no hot path
vec[0] = amount / norm.MaxAmount

// DEPOIS: multiplicação por inverso pré-computado
vec[0] = amount * invMaxAmount
```

### 2. Parser JSON Manual

- Pula campos irrelevantes sem alocar strings
- `parseFloatFast` evita `strconv.ParseFloat`
- `parseIntFast` evita `strconv.ParseInt`
- `parseJSONStringBytes` extrai bytes sem alocar string

---

## Otimizações de Memória

### 1. SetMemoryLimit

```go
// cmd/api/main.go
debug.SetMemoryLimit(150 << 20)  // 150 MiB
```

Força o GC a ser mais agressivo antes de atingir o limite de 170 MB.

### 2. Background Page Touching

```go
// internal/knn/grid.go
go func() {
    ticker := time.NewTicker(10 * time.Second)
    defer ticker.Stop()
    for range ticker.C {
        for i := 0; i < len(vectors); i += 2048 {
            _ = vectors[i]  // Touch cada página
        }
    }
}()
```

Mantém páginas do grid.bin residentes em memória, evitando page faults.

### 3. Quantização int16

| Formato | Tamanho (3M × 14 dims) |
|---------|------------------------|
| float32 | 168 MB |
| int16 | 84 MB |
| int16 + labels bit-packed | ~84.4 MB |

---

## Benchmarks

### Grid Index PredictRaw

```
BenchmarkGridIndex_PredictRaw-16    32660    115002 ns/op    0 B/op    0 allocs/op
```

### Comparação com IVF

| Métrica | Grid Index | IVF Index |
|---------|-----------|-----------|
| Latência (microbenchmark) | ~115μs | ~19-21μs |
| Precisão | Exata | Aproximada |
| Alocações | 0 | 0 |
| Tamanho do índice | ~88 MB | ~95 MB |

> Nota: O Grid Index tem latência maior por ser exato (busca em todas as partições relevantes), mas garante FP=0.

---

## Profiling

### CPU Profile

```bash
go test -bench=BenchmarkGridIndex_PredictRaw -cpuprofile=cpu.prof ./internal/knn/
go tool pprof --text cpu.prof
```

Hot path típico:
1. `gridSearchKDTree` — travessia da KD-tree
2. `gridScanLeaf` — scanning de folhas com AVX2
3. `gridBBoxLowerBound` — bounds computation
4. `quantizeQuery` — quantização da query

### Memory Profile

```bash
go test -bench=BenchmarkGridIndex_PredictRaw -memprofile=mem.prof ./internal/knn/
go tool pprof --text mem.prof
```

Resultado: 0 B/op, 0 allocs/op no hot path.

---

## Configurações de Runtime

| Variável | Valor | Motivo |
|----------|-------|--------|
| `GOMAXPROCS` | 1 (por container) | Evita overhead de scheduling |
| `GOGC` | off (via SetMemoryLimit) | Controle fino de memória |
| `debug.SetMemoryLimit` | 150 MiB | GC agressivo antes do limite |
| `TELEMETRY_ENABLED` | false | Sem overhead de logging |

---

## Checklist de Otimizações

- [x] Zero alocações no hot path
- [x] Respostas pré-computadas
- [x] Parser HTTP manual
- [x] Parser JSON manual
- [x] Multiplicação por inverso
- [x] Quantização int16
- [x] SIMD AVX2 com acumuladores paralelos
- [x] Bounds pruning
- [x] Early exit (bounds, worstDist=0)
- [x] sync.Pool para buffers
- [x] Telemetria lock-free
- [x] Lookup O(1) de partição
- [x] gridBBoxLowerBound desenrolado
- [x] Aritmética int16 no bounds
- [x] PREFETCHT0 para próximo bloco
- [x] Background page touching
- [x] SetMemoryLimit para controle de GC
