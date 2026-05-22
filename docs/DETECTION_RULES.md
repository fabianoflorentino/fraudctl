# Regras de Detecção

## Visão Geral

O backend transforma a transação em um vetor de 14 dimensões, busca os 5 vizinhos mais próximos no índice Grid (KD-tree) e decide a aprovação com base na contagem de vizinhos fraudados.

## Regra de Decisão

1. vetor da transação
2. `PredictRaw` retorna `fraudCount` entre `0` e `5`
3. `fraud_score = fraudCount / 5`
4. `approved = fraudCount < 3`

## Fluxo de Decisão

```mermaid
flowchart TD
    R[Request JSON] --> V[Vectorizer]
    V --> Q[Vector14]
    Q --> K[GridIndex PredictRaw]
    K --> F[fraudCount 0 to 5]
    F --> S[fraud_score = fraudCount / 5]
    F --> A[approved quando fraudCount < 3]
    S --> O[Response JSON]
    A --> O
```

## Dimensões do Vetor

| índice | feature | regra |
|---|---|---|
| 0 | `amount` | `clamp(amount / max_amount)` |
| 1 | `installments` | `clamp(installments / max_installments)` |
| 2 | `amount_vs_avg` | `clamp((amount / customer.avg_amount) / amount_vs_avg_ratio)` |
| 3 | `hour_of_day` | `hour(requested_at) / 23` |
| 4 | `day_of_week` | `weekday(requested_at) / 6` com `mon=0` |
| 5 | `minutes_since_last_tx` | `clamp(minutes / max_minutes)` ou `-1` |
| 6 | `km_from_last_tx` | `clamp(km / max_km)` ou `-1` |
| 7 | `km_from_home` | `clamp(km_from_home / max_km)` |
| 8 | `tx_count_24h` | `clamp(tx_count_24h / max_tx_count_24h)` |
| 9 | `is_online` | `1` ou `0` |
| 10 | `card_present` | `1` ou `0` |
| 11 | `unknown_merchant` | `1` se `merchant.id` não estiver em `known_merchants` |
| 12 | `mcc_risk` | valor de `mcc_risk.json`, default `0.5` |
| 13 | `merchant_avg_amount` | `clamp(merchant.avg_amount / max_merchant_avg_amount)` |

## Regras Especiais

### `last_transaction == null`

Quando não existe transação anterior:

- `vec[5] = -1`
- `vec[6] = -1`

Esse é o único caso em que o vetor pode conter valor fora de `[0,1]`.

### Merchant conhecido

`unknown_merchant` é `0` quando `merchant.id` aparece em `customer.known_merchants`.

### MCC risk

- lookup em estrutura array de `10000` posições
- MCC inválido ou ausente cai em `0.5`

## Normalização

Os limites vêm de `resources/normalization.json`.

Campos usados:

- `max_amount`
- `max_installments`
- `amount_vs_avg_ratio`
- `max_minutes`
- `max_km`
- `max_tx_count_24h`
- `max_merchant_avg_amount`

`clamp(x)` limita o valor ao intervalo `[0.0, 1.0]`.

## Busca KNN Atual (Grid Index)

Implementação atual em `internal/knn/grid.go`.

Passos:

1. quantiza query para `int16` com escala `10000`
2. calcula chave de partição de 8 bits baseada nas features
3. lookup direto da partição via `partitionsByKey[key]` (O(1))
4. calcula lower bound da bounding box da partição com aritmética int16
5. se lower bound < worstDist, traversa a KD-tree da partição:
   - split na dimensão mais larga (median split)
   - leaf size: 128 vetores
   - bounds pruning em cada nó
   - scanning de folhas com kernel AVX2 (`ScanBlock8AVX2`)
6. early exit se worstDist = 0 (match exato)
7. se ainda houver budget, scan de outras partições com bounds pruning
8. retorna a contagem final de vizinhos fraudados

```mermaid
flowchart TD
    Q[Query float32 14] --> QQ[Quantize int16]
    QQ --> PK[Partition Key 8 bits]
    PK --> PL[Direct Lookup partitionsByKey]
    PL --> BB[BBox Lower Bound int16]
    BB --> KD{KD-tree traversal}
    KD -->|leaf| AVX[ScanBlock8AVX2]
    KD -->|internal| BP[Bounds pruning]
    AVX --> WD{worstDist == 0?}
    WD -->|sim| RET[Return fraudCount]
    WD -->|não| OP[Other partitions scan]
    BP --> KD
    OP --> RET
```

## Chave de Partição (8 bits)

| Bit | Feature | Condição |
|-----|---------|----------|
| 0 | last_tx | dim[5] >= 0 |
| 1 | is_online | dim[9] > 0 |
| 2 | card_present | dim[10] > 0 |
| 3 | unknown_merchant | dim[11] > 0 |
| 4-5 | mcc_bucket | dim[12] (0-3) |
| 6 | amount_high | dim[2] > 5000 |
| 7 | tx_count_high | dim[8] > 2500 |

## SIMD via Go Assembly (amd64)

Em CPUs amd64 com AVX2, o kernel `ScanBlock8AVX2` acelera o scanning de folhas:

- Processa 8 vetores simultaneamente
- Early exit na dim 8 se todos os 8 vetores excedem worstDist
- Acumuladores paralelos (pares e ímpares) para reduzir dependency chain
- PREFETCHT0 para prefetch do próximo bloco

## Observações Importantes do Estado Atual

- O handler atual usa apenas KNN para a decisão.
- Embora `dataset` consiga carregar `model.bin` e `gbdt.bin`, esse resultado não é usado no caminho principal atual da API.
- O parser de JSON do hot path é manual e orientado a bytes.
- O índice IVF (`ivf.bin`) é fallback quando `grid.bin` não está disponível.
