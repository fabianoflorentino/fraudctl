---
title: "Detecção"
description: "As 14 dimensões, a normalização, a decisão K=5 e como a chave de partição reduz o espaço de busca."
weight: 30
---

A detecção é uma busca pelos **5 vizinhos mais próximos** em um espaço de 14 dimensões. A decisão é a fração de vizinhos rotulados como fraude.

## As 14 dimensões

| # | Feature | Regra |
|---|---|---|
| 0 | `amount` | `clamp(amount / max_amount)` |
| 1 | `installments` | `clamp(installments / max_installments)` |
| 2 | `amount_vs_avg` | `clamp((amount / customer.avg_amount) / amount_vs_avg_ratio)` |
| 3 | `hour_of_day` | `hour(requested_at) / 23` |
| 4 | `day_of_week` | `weekday(requested_at) / 6`, com segunda = 0 |
| 5 | `minutes_since_last_tx` | `clamp(minutes / max_minutes)` ou `-1` |
| 6 | `km_from_last_tx` | `clamp(km / max_km)` ou `-1` |
| 7 | `km_from_home` | `clamp(km_from_home / max_km)` |
| 8 | `tx_count_24h` | `clamp(tx_count_24h / max_tx_count_24h)` |
| 9 | `is_online` | `1` ou `0` |
| 10 | `card_present` | `1` ou `0` |
| 11 | `unknown_merchant` | `1` se o comerciante não está na lista de conhecidos |
| 12 | `mcc_risk` | valor de `mcc_risk.json`, default `0,5` |
| 13 | `merchant_avg_amount` | `clamp(merchant.avg_amount / max_merchant_avg_amount)` |

`clamp(x)` limita o valor ao intervalo `[0, 1]`. Quando não existe transação anterior, as dimensões 5 e 6 valem `-1` — o único caso de valor fora do intervalo.

## Vetorização sem alocação

O `VectorizeJSON` não desserializa a request inteira. O parser percorre os bytes uma única vez e extrai apenas os campos usados, com `parseFloatFast`, `parseIntFast` e extração de string sem alocar. As constantes de normalização são multiplicadas por inversos pré-computados, evitando divisão no hot path.

## KNN exato

O índice é uma **KD-tree particionada**:

1. a query é quantizada para `int16` com escala `10000`;
2. uma chave de 8 bits seleciona a partição, com lookup O(1) em `partitionsByKey[256]`;
3. a bounding box da partição é comparada com o pior candidato atual;
4. a árvore é percorrida com poda por bounds;
5. folhas são varridas com o kernel AVX2, 8 vetores por bloco;
6. se a pior distância chega a zero, a busca termina imediatamente.

### Chave de partição

| Bit | Feature | Condição |
|---|---|---|
| 0 | `last_tx` | `dim[5] >= 0` |
| 1 | `is_online` | `dim[9] > 0` |
| 2 | `card_present` | `dim[10] > 0` |
| 3 | `unknown_merchant` | `dim[11] > 0` |
| 4–5 | `mcc_bucket` | `dim[12]` (0–3) |
| 6 | `amount_high` | `dim[2] > 5000` |
| 7 | `tx_count_high` | `dim[8] > 2500` |

## Por que exato importa

Um índice aproximado pode errar a lista dos 5 vizinhos e, com isso, produzir um falso positivo ou falso negativo. Como a componente de detecção do score penaliza erros e zera em 15% de falhas, a versão final priorizou **exatidão** com poda agressiva em vez de aproximação.
