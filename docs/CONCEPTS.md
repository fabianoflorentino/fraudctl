# Conceitos e Técnicas do Projeto

Este documento explica todos os conceitos, técnicas e padrões usados no projeto fraudctl, com exemplos de código.

## Índice

1. [Visão Geral da Arquitetura](#visão-geral-da-arquitetura)
2. [Servidor HTTP Custom (rawhttp)](#servidor-http-custom-rawhttp)
3. [FD Passing via SCM_RIGHTS](#fd-passing-via-scm_rights)
4. [Vectorização de Features](#vectorização-de-features)
5. [Parser JSON Manual](#parser-json-manual)
6. [Grid Index (KD-tree)](#grid-index-kd-tree)
7. [IVF Index (Fallback)](#ivf-index-fallback)
8. [Brute Force Index (Fallback)](#brute-force-index-fallback)
9. [SIMD com Go Assembly](#simd-com-go-assembly)
10. [Top-K com Inserção Cascata](#top-k-com-inserção-cascata)
11. [Quantização int16](#quantização-int16)
12. [Bounds Pruning](#bounds-pruning)
13. [Zero-Allocation Hot Path](#zero-allocation-hot-path)
14. [sync.Pool para Buffers](#syncpool-para-buffers)
15. [Telemetria Lock-Free](#telemetria-lock-free)
16. [Respostas Pré-computadas](#respostas-pré-computadas)

---

## Visão Geral da Arquitetura

O fraudctl é uma API de detecção de fraude que usa busca KNN (k-nearest neighbors) sobre 3 milhões de vetores de referência.

```
Cliente → fraudctl-lb (TCP :9999) → API-1 ou API-2 → GridIndex → JSON Response
```

**Componentes principais:**

| Componente | Pacote | Responsabilidade |
|------------|--------|------------------|
| Load Balancer | `cmd/lb` | Aceita TCP, passa FD via SCM_RIGHTS |
| API | `cmd/api` | Carrega dataset, serve conexões |
| HTTP Server | `internal/rawhttp` | Parser HTTP manual, zero alocação |
| Handler | `internal/handler` | Orquestra vectorização + KNN |
| Vectorizer | `internal/vectorizer` | JSON → Vector14 |
| Grid Index | `internal/knn/grid.go` | KD-tree exato com AVX2 |
| IVF Index | `internal/knn/brute.go` | Fallback aproximado |
| Dataset | `internal/dataset` | Carrega índices e configurações |

---

## Servidor HTTP Custom (rawhttp)

**Problema:** `net/http` e `fasthttp` têm overhead desnecessário para nosso caso de uso simples (apenas 2 endpoints, sempre 200 OK).

**Solução:** Parser HTTP manual orientado a bytes.

```go
// internal/rawhttp/server.go

func (s *Server) ServeConn(conn net.Conn) error {
    bp := bufPool.Get().(*[]byte)    // Buffer do pool (1 KiB)
    buf := *bp
    pos := 0
    used := 0
    defer bufPool.Put(bp)

    for {
        // Lê headers até encontrar \r\n\r\n
        headEnd := indexHeaderEnd(buf[pos:used])
        for headEnd < 0 {
            n, err := conn.Read(buf[used:])
            used += n
            headEnd = indexHeaderEnd(buf[pos:used])
        }

        // Parse da request line
        method, path, contentLen := parseRequestLine(buf[pos:headEnd])

        // Lê body se necessário
        bodyEnd := headEnd + contentLen
        for used < bodyEnd {
            n, _ := conn.Read(buf[used:])
            used += n
        }

        // Roteamento por bytes (sem string allocation)
        var resp []byte
        switch {
        case len(path) == 12 && path[1] == 'f':  // /fraud-score
            resp = s.handler.ServeFraudScore(buf[headEnd:bodyEnd])
        case len(path) == 6 && path[1] == 'r':   // /ready
            resp = s.handler.ServeReady()
        default:
            resp = notFoundResponse
        }

        conn.Write(resp)

        // Reutiliza buffer para próxima request (pipelining)
        pos = bodyEnd
        if pos >= used {
            pos = 0
            used = 0
        }
    }
}
```

**Por que funciona:**
- Buffer fixo de 1 KiB via `sync.Pool` (zero alocação por request)
- Roteamento por comparação de bytes (sem `string()` conversion)
- Suporte a HTTP pipelining (múltiplas requests na mesma conexão)
- Read deadline de 10s, write deadline de 750ms

---

## FD Passing via SCM_RIGHTS

**Problema:** O load balancer precisa transferir conexões TCP para as APIs sem proxy HTTP.

**Solução:** Unix domain sockets com `SCM_RIGHTS` para passar file descriptors.

```go
// cmd/lb/main.go — Lado do LB

func passFD(conn net.TCPConn, workerUnix *net.UnixConn) {
    f, _ := conn.File()
    fd := int(f.Fd())

    // SCM_RIGHTS layout: cmsghdr (16B) + fd (4B) = 20B
    oob := syscall.UnixRights(fd)
    workerUnix.WriteMsgUnix([]byte{0}, oob, nil)

    f.Close()  // LB fecha sua cópia, API agora é dona do FD
}

// cmd/api/main.go — Lado da API

func serveControl(ctrlConn net.Conn, srv *rawhttp.Server) {
    uc := ctrlConn.(*net.UnixConn)
    oob := make([]byte, 64)

    for {
        _, oobn, _, _, err := uc.ReadMsgUnix(buf, oob)
        fd := parseUnixRights(oob[:oobn])  // Parse manual, zero alloc

        file := os.NewFile(uintptr(fd), "")
        conn, _ := net.FileConn(file)  // Reconstrói net.Conn do FD
        go srv.ServeConn(conn)
    }
}
```

**Por que funciona:**
- Zero cópia de dados (o FD aponta para o mesmo socket kernel)
- Parse manual de `cmsghdr` para evitar alocação de `ParseSocketControlMessage`
- A API é dona do FD após recebimento (sem race conditions)

---

## Vectorização de Features

**Problema:** Converter JSON de transação em vetor de 14 dimensões normalizadas.

**Solução:** 14 features com normalização por constantes pré-carregadas.

```go
// internal/vectorizer/vectorizer.go

type Vectorizer struct {
    invMaxAmount   float32  // 1.0 / norm.MaxAmount
    invMaxInstall  float32  // 1.0 / norm.MaxInstallments
    invMaxMinutes  float32
    invMaxKM       float32
    invMaxTxCount  float32
    invAmountRatio float32
    invMerchantAvg float32
    mccRisk        model.MCCRisk  // Array de 10000 posições
}

func (v *Vectorizer) Vectorize(req *FraudScoreRequest) model.Vector14 {
    var vec model.Vector14

    // dim 0: amount normalizado
    vec[0] = clampFloat32(float32(req.Transaction.Amount) * v.invMaxAmount)

    // dim 1: parcelas normalizadas
    vec[1] = clampFloat32(float32(req.Transaction.Installments) * v.invMaxInstall)

    // dim 2: amount vs média do cliente
    avgRatio := req.Transaction.Amount / req.Customer.AvgAmount
    vec[2] = clampFloat32(float32(avgRatio) * v.invAmountRatio)

    // dim 3-4: hora e dia da semana
    t := req.Transaction.RequestedAtTime()
    vec[3] = float32(t.Hour()) / 23.0
    vec[4] = float32(t.Weekday()) / 6.0

    // dim 5-6: minutos e km da última transação (-1 se null)
    if req.LastTransaction == nil {
        vec[5] = -1
        vec[6] = -1
    } else {
        minutes := req.RequestedAtTime().Sub(req.LastTransaction.TimestampTime()).Minutes()
        vec[5] = clampFloat32(float32(minutes) * v.invMaxMinutes)
        vec[6] = clampFloat32(float32(req.LastTransaction.KmFromCurrent) * v.invMaxKM)
    }

    // dim 7: km da casa
    vec[7] = clampFloat32(float32(req.Terminal.KmFromHome) * v.invMaxKM)

    // dim 8: transações nas últimas 24h
    vec[8] = clampFloat32(float32(req.Customer.TxCount24h) * v.invMaxTxCount)

    // dim 9-11: flags binárias
    vec[9] = boolFloat32(req.Terminal.IsOnline)
    vec[10] = boolFloat32(req.Terminal.CardPresent)
    vec[11] = boolFloat32(!v.isKnownMerchant(req.Merchant.ID, req.Customer.KnownMerchants))

    // dim 12: MCC risk score
    vec[12] = v.mccRisk.Get(req.Merchant.MCC)

    // dim 13: média do merchant
    vec[13] = clampFloat32(float32(req.Merchant.AvgAmount) * v.invMerchantAvg)

    return vec
}
```

**Otimização:** Multiplicação por inverso pré-computado ao invés de divisão no hot path.

---

## Parser JSON Manual

**Problema:** `json.Unmarshal` aloca a struct inteira mesmo quando só precisamos de alguns campos.

**Solução:** Parser byte-oriented que extrai apenas os valores necessários.

```go
// internal/vectorizer/fast_json.go

func (v *Vectorizer) VectorizeJSON(data []byte) (model.Vector14, error) {
    var vec model.Vector14

    // Estado do parser
    var (
        amount       float64
        installments int64
        // ... outros campos
    )

    // Scan top-level e nested objects
    // Para cada campo relevante, parse direto de bytes → float64/int64
    // Exemplo: "amount": 384.88 → scan até ':', depois parseFloatFast

    // parseFloatFast: converte "384.88" → 384.88 sem strconv.ParseFloat
    func parseFloatFast(b []byte) float64 {
        var result float64
        var frac float64
        var fracDiv float64 = 1
        negative := false
        inFraction := false

        for _, c := range b {
            switch c {
            case '-':
                negative = true
            case '.':
                inFraction = true
            default:
                digit := float64(c - '0')
                if inFraction {
                    fracDiv *= 10
                    frac += digit / fracDiv
                } else {
                    result = result*10 + digit
                }
            }
        }

        result += frac
        if negative {
            result = -result
        }
        return result
    }
}
```

**Por que funciona:**
- Pula campos irrelevantes sem alocar strings
- `parseFloatFast` evita `strconv.ParseFloat` (mais rápido para casos comuns)
- Zero alocação no hot path

---

## Grid Index (KD-tree)

**Problema:** Buscar 5 vizinhos mais próximos em 3M vetores com latência < 1ms.

**Solução:** KD-tree com partições de 8 bits e scanning AVX2.

### Estrutura do Índice

```
grid.bin:
┌─────────────────────────────────────┐
│ Header: magic + version + n + dim   │ 24 bytes
├─────────────────────────────────────┤
│ Partitions: key + root + bbox       │ ~192 × 80 bytes
├─────────────────────────────────────┤
│ Nodes: left + right + block + bbox  │ ~61K × 68 bytes
├─────────────────────────────────────┤
│ Vectors: block-major int16          │ 3M × 14 × 2 bytes
├─────────────────────────────────────┤
│ Labels: bit-packed                  │ 3M / 8 bytes
└─────────────────────────────────────┘
```

### Chave de Partição (8 bits)

| Bit | Feature | Condição |
|-----|---------|----------|
| 0 | last_tx | dim[5] >= 0 |
| 1 | is_online | dim[9] > 0 |
| 2 | card_present | dim[10] > 0 |
| 3 | unknown_merchant | dim[11] > 0 |
| 4-5 | mcc_bucket | dim[12] (0-3) |
| 6 | amount_high | dim[2] > 5000 |
| 7 | tx_count_high | dim[8] > 2500 |

### Busca

```go
// internal/knn/grid.go

func (idx *GridIndex) PredictRaw(query model.Vector14, _ int) int {
    qi := quantizeQuery(query)           // float32 → int16 (escala 10000)
    key := gridPartitionKey(&qi)         // Calcula chave de 8 bits

    h := newTopK5()                      // Top-5 sorted array

    // 1. Lookup O(1) da partição correspondente
    sameIdx := idx.partitionsByKey[key]

    if sameIdx >= 0 {
        p := &idx.partitions[sameIdx]
        lb := gridBBoxLowerBound(&qi, &p.bboxMin, &p.bboxMax)
        if lb < h.worstDist() {
            gridSearchKDTree(p.root, lb, &qi, &h, idx)
        }
        if h.worstDist() == 0 {
            return h.fraudCount(idx.labels)  // Early exit: match exato
        }
    }

    // 2. Scan de outras partições com bounds pruning
    for i := range idx.partitions {
        if int32(i) == sameIdx { continue }
        p := &idx.partitions[i]

        // Bounds computation com aritmética int16 (sem f32)
        var lb uint64
        worst := h.worstDist()
        for d := 0; d < 14; d++ {
            qd := qi[d]
            if qd < p.bboxMin[d] {
                diff := int32(p.bboxMin[d]) - int32(qd)
                lb += uint64(diff) * uint64(diff)
            } else if qd > p.bboxMax[d] {
                diff := int32(qd) - int32(p.bboxMax[d])
                lb += uint64(diff) * uint64(diff)
            }
            if lb >= worst { break }  // Early exit
        }
        if lb < worst {
            gridSearchKDTree(p.root, lb, &qi, &h, idx)
        }
    }

    return h.fraudCount(idx.labels)
}
```

### KD-tree Traversal

```go
func gridSearchKDTree(root int32, rootBound uint64, qi *[DIM]int16, h *topK5, idx *GridIndex) {
    stackLen := 0
    current := root
    currentBound := rootBound

    for {
        if currentBound < h.worstDist() {
            n := &idx.nodes[current]
            if n.left < 0 || n.right < 0 {
                // Nó folha: scan com AVX2
                gridScanLeaf(idx.vectors, idx.labels, int(n.blockStart),
                    int(n.vecOffset), int(n.blockLen), *qi, h, &idx.scratch.blockSum)
            } else {
                // Nó interno: calcula bounds dos filhos
                lNode := &idx.nodes[n.left]
                rNode := &idx.nodes[n.right]
                lb := gridBBoxLowerBound(qi, &lNode.bboxMin, &lNode.bboxMax)
                rb := gridBBoxLowerBound(qi, &rNode.bboxMin, &rNode.bboxMax)

                // Visita o filho mais próximo primeiro
                if lb <= rb {
                    nearIdx, nearBound = n.left, lb
                    farIdx, farBound = n.right, rb
                } else {
                    nearIdx, nearBound = n.right, rb
                    farIdx, farBound = n.left, lb
                }

                // Empilha filho distante se puder melhorar
                worst := h.worstDist()
                if farBound < worst && stackLen < 64 {
                    idx.scratch.stackNodes[stackLen] = farIdx
                    idx.scratch.stackBounds[stackLen] = farBound
                    stackLen++
                }

                // Desce para filho próximo
                if nearBound < worst {
                    current = nearIdx
                    currentBound = nearBound
                    continue
                }
            }
        }

        // Pop da stack
        if stackLen == 0 { return }
        stackLen--
        current = idx.scratch.stackNodes[stackLen]
        currentBound = idx.scratch.stackBounds[stackLen]
    }
}
```

---

## IVF Index (Fallback)

Quando `grid.bin` não está disponível, o IVF Index é usado como fallback.

**Conceito:** Inverted File Index — divide vetores em clusters (Voronoi) e busca apenas nos clusters mais próximos.

```go
// internal/knn/brute.go

type IVFIndex struct {
    centroids     []float32  // SoA: centroids[d*nlist + ci]
    centroidNorms []float32  // ||c||² pré-computado
    vectors       []int16    // AoS: [N * DIM]
    offsets       []uint32   // offsets[ci]..offsets[ci+1] = vetores do cluster
    bboxMin       []int16    // Bounding box por cluster
    bboxMax       []int16
    nlist         int        // Número de clusters
    nprobe        int        // Clusters a visitar
    quickProbe    int        // Quick probe (1ª passada)
    boundaryLo    int        // Zona ambígua inferior
    boundaryHi    int        // Zona ambígua superior
}
```

**Busca 2-tier:**
1. Quick probe: escaneia `quickProbe` clusters
2. Se fraudCount ∈ [boundaryLo, boundaryHi], expande para `nprobe` clusters

---

## Brute Force Index (Fallback)

Quando nenhum índice pré-construído está disponível.

```go
// internal/knn/brute_pure.go

type BruteIndex struct {
    flat       []float32  // Vetores em array flat
    fraudFlags []bool     // Labels
    count      int
}

// Busca exata: calcula distância para TODOS os vetores
func brutePredict(flat []float32, fraudFlags []bool, count int, query model.Vector14, k int) float64 {
    // Mantém top-K em array sorted
    // Para cada vetor: dist = Σ(q[i] - v[i])²
    // Se dist < worstDist, insere no top-K
}
```

---

## SIMD com Go Assembly

**Problema:** Calcular distância L2 entre query e 8 vetores simultaneamente.

**Solução:** `ScanBlock8AVX2` em Go Assembly usando AVX2.

```asm
; internal/knn/grid_kernel.s
; ScanBlock8AVX2(q *float32, block *int16, worst float32, sum *[8]float32) bool

TEXT ·ScanBlock8AVX2(SB), NOSPLIT, $0-33
    MOVQ         q+0(FP), AX
    MOVQ         block+8(FP), BX
    VBROADCASTSS worst+16(FP), Y15

    ; Prefetch do próximo bloco
    PREFETCHT0 256(BX)
    PREFETCHT0 320(BX)
    PREFETCHT0 384(BX)

    ; Dois acumuladores paralelos (pares e ímpares)
    VXORPS Y0, Y0, Y0     ; even-dim acc
    VXORPS Y4, Y4, Y4     ; odd-dim acc

    ; Dim 0 → Y0 (even)
    VPMOVSXWD    (BX), Y1      ; int16×8 → int32×8
    VCVTDQ2PS    Y1, Y1        ; int32×8 → float32×8
    VBROADCASTSS (AX), Y2      ; query[0] em 8 lanes
    VSUBPS       Y1, Y2, Y2    ; diff = q - v
    VFMADD231PS  Y2, Y2, Y0    ; acc += diff²

    ; Dim 1 → Y4 (odd)
    VPMOVSXWD    16(BX), Y1
    ...

    ; Checkpoint na dim 8: early exit
    VADDPS    Y4, Y0, Y3      ; Merge accs
    VCMPPS    $0x01, Y15, Y3, Y3  ; compare < worst
    VMOVMSKPS Y3, R8
    TESTL     R8, R8
    JZ        dead             ; Todos excedem worst → skip

    ; Continua dims 8-13...
    ; Final merge + store
    VMOVUPS Y0, (CX)
    MOVB    $1, ret+32(FP)
    RET
```

**Otimizações:**
- **Acumuladores paralelos:** Y0 (dims pares) e Y4 (dims ímpares) rodam em pipelines independentes
- **Early exit na dim 8:** Se todos os 8 vetores excedem `worst`, pula o bloco
- **PREFETCHT0:** Traz próximo bloco para L1 enquanto FMAs estão em flight
- **VFMADD231PS:** Fused multiply-add (diff × diff + acc) em uma instrução

---

## Top-K com Inserção Cascata

**Problema:** Manter os 5 menores distâncias sem heap ou alocação.

**Solução:** Array sorted de tamanho fixo com inserção cascata.

```go
// internal/knn/ivf_search.go

type topK5 struct {
    dist  [K]uint64  // Distâncias ordenadas (menor primeiro)
    idx   [K]int     // Índices dos vetores
    count int        // Quantos slots preenchidos
}

//go:inline
func (h *topK5) worstDist() uint64 { return h.dist[K-1] }

//go:inline
func (h *topK5) tryInsert(dist uint64, vecIdx int) {
    if dist >= h.worstDist() {
        return  // Não é melhor que o pior atual
    }

    // Encontra posição de inserção (desenrolado para K=5)
    i := h.count
    if i == K {
        i = K - 1
    }
    for i > 0 && dist < h.dist[i-1] {
        h.dist[i] = h.dist[i-1]  // Shift para direita
        h.idx[i] = h.idx[i-1]
        i--
    }
    h.dist[i] = dist
    h.idx[i] = vecIdx
    if h.count < K {
        h.count++
    }
}
```

**Por que funciona:**
- Array fixo de 5 elementos (sem alocação)
- `//go:inline` elimina overhead de chamada
- Early return se dist >= worstDist
- Shift cascata é rápido para K=5 (máx 4 moves)

---

## Quantização int16

**Problema:** Vetores float32 ocupam 4 bytes por dimensão. 3M × 14 × 4 = 168 MB.

**Solução:** Quantizar para int16 com escala 10000.

```go
const int16Scale = 10000

func quantizeFloat32(v float32) int16 {
    return int16(v * int16Scale)
}

func quantizeQuery(q model.Vector14) [DIM]int16 {
    var qi [DIM]int16
    for d := 0; d < DIM; d++ {
        qi[d] = int16(q[d] * int16Scale)
    }
    return qi
}
```

**Precisão:**
- float32: ~7 dígitos decimais
- int16 com escala 10000: 4 casas decimais (suficiente para features [0,1])
- Economia: 50% de memória (2 bytes vs 4 bytes por dimensão)

---

## Bounds Pruning

**Problema:** Não adianta escanear um cluster/partição se a bounding box está longe demais.

**Solução:** Calcular lower bound da distância entre query e bounding box.

```go
// gridBBoxLowerBound desenrolado para 14 dimensões
func gridBBoxLowerBound(q *[DIM]int16, bmin, bmax *[DIM]int16) uint64 {
    var d uint64

    qi0 := int32(q[0])
    if qi0 < int32(bmin[0]) {
        diff := int32(bmin[0]) - qi0
        d += uint64(diff) * uint64(diff)
    } else if qi0 > int32(bmax[0]) {
        diff := qi0 - int32(bmax[0])
        d += uint64(diff) * uint64(diff)
    }
    // ... repetido para dims 1-13

    return d
}
```

**Como funciona:**
- Se query está dentro da bbox em todas as dimensões → lower bound = 0
- Se query está fora em alguma dimensão → lower bound = Σ dist²
- Se lower bound >= worstDist → skip (nenhum vetor na bbox pode ser melhor)

**Otimização:** Desenrolado para 14 dimensões permite que o compilador otimize melhor (sem loop overhead, melhor register allocation).

---

## Zero-Allocation Hot Path

**Objetivo:** Zero alocações no caminho de processamento de cada request.

**Técnicas usadas:**

| Técnica | Onde | Impacto |
|---------|------|---------|
| `sync.Pool` para buffers | `rawhttp` | Evita alloc de 1 KiB por request |
| Array fixo para top-K | `knn/topK5` | Evita alloc de heap |
| scratch reutilizável | `GridIndex.scratch` | Evita alloc de stack nodes |
| Respostas pré-computadas | `rawhttp/init()` | Evita alloc de JSON por request |
| Parser JSON manual | `vectorizer/fast_json.go` | Evita alloc de struct |
| Parse manual de SCM_RIGHTS | `cmd/api/main.go` | Evita alloc de `ParseSocketControlMessage` |

**Verificação:**
```bash
go test -bench=. -benchmem ./internal/knn/
# Output: 0 B/op, 0 allocs/op
```

---

## sync.Pool para Buffers

```go
// internal/rawhttp/server.go

var bufPool = sync.Pool{
    New: func() any {
        b := make([]byte, readBufSize)  // 1 KiB
        return &b
    },
}

func (s *Server) ServeConn(conn net.Conn) error {
    bp := bufPool.Get().(*[]byte)
    buf := *bp
    defer bufPool.Put(bp)  // Devolve ao pool

    // Usa buf para ler e parsear HTTP...
}
```

**Por que funciona:**
- Buffer é reutilizado entre requests
- Evita pressão no GC
- `defer` garante devolução mesmo em caso de erro

---

## Telemetria Lock-Free

**Problema:** Coletar métricas sem adicionar overhead ao hot path.

**Solução:** Contadores atômicos + ring buffer para outliers.

```go
// internal/middleware/logger.go

type Stats struct {
    total        int64
    errors       int64
    fraud        int64
    legit        int64
    borderZone   int64  // fraudCount ∈ {2,3}
    parseErrors  int64
    totalLatency int64
    buckets      [numBuckets]int64  // Histograma de latência
    _            [8]int64           // Cache-line padding
}

var global Stats

func Record(elapsed time.Duration, ...) {
    atomic.AddInt64(&global.total, 1)
    atomic.AddInt64(&global.totalLatency, int64(elapsed))

    // Histogram bucket
    us := elapsed.Microseconds()
    for i, bound := range bucketBounds {
        if us <= bound {
            atomic.AddInt64(&global.buckets[i], 1)
            break
        }
    }

    // Outlier capture (apenas requests > 2ms)
    if us > outlierThresholdUs {
        storeOutlier(elapsed, fraudCount, ...)
    }
}
```

**Por que funciona:**
- `atomic.AddInt64` é lock-free (instrução CPU `LOCK XADD`)
- Cache-line padding evita false sharing entre cores
- Outliers capturados em ring buffer lock-free (atomic CAS)
- Reporter periódico drena e imprime stats

---

## Respostas Pré-computadas

**Problema:** Serializar JSON para cada resposta desperdiça CPU.

**Solução:** Pré-computar todas as 6 respostas possíveis no `init()`.

```go
// internal/rawhttp/server.go

var fraudResponses [6][]byte

func init() {
    bodies := [6]string{
        `{"approved":true,"fraud_score":0.0}`,
        `{"approved":true,"fraud_score":0.2}`,
        `{"approved":true,"fraud_score":0.4}`,
        `{"approved":false,"fraud_score":0.6}`,
        `{"approved":false,"fraud_score":0.8}`,
        `{"approved":false,"fraud_score":1.0}`,
    }

    for i, body := range bodies {
        fraudResponses[i] = buildOKResponse("application/json", body)
    }
}

func FraudResponse(count int) []byte {
    if count < 0 || count >= len(fraudResponses) {
        return fraudResponses[0]
    }
    return fraudResponses[count]
}
```

**Resultado:**
- `buildOKResponse` monta HTTP completo: `HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 40\r\n\r\n{"approved":true,"fraud_score":0.0}`
- No hot path: apenas retorno de slice pré-computado (zero CPU)
