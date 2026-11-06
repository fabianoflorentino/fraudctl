---
title: "Detection"
description: "Decision model, dimensions, search process, and quality controls in fraudctl."
weight: 30
---

## Decision model

The decision is based on a search for the **5 nearest neighbors** in a 14-dimensional space. `fraud_score` is the fraction of neighbors labeled as fraud, and approval occurs when the score is below 0.6.

This approach preserves traceability: the decision can be explained by the neighborhood found, which supports auditing and case analysis.

## Vector dimensions

| # | Feature | Rule |
|---|---|---|
| 0 | `amount` | `clamp(amount / max_amount)` |
| 1 | `installments` | `clamp(installments / max_installments)` |
| 2 | `amount_vs_avg` | `clamp((amount / customer.avg_amount) / amount_vs_avg_ratio)` |
| 3 | `hour_of_day` | `hour(requested_at) / 23` |
| 4 | `day_of_week` | `weekday(requested_at) / 6`, with Monday = 0 |
| 5 | `minutes_since_last_tx` | `clamp(minutes / max_minutes)` or `-1` |
| 6 | `km_from_last_tx` | `clamp(km / max_km)` or `-1` |
| 7 | `km_from_home` | `clamp(km_from_home / max_km)` |
| 8 | `tx_count_24h` | `clamp(tx_count_24h / max_tx_count_24h)` |
| 9 | `is_online` | `1` or `0` |
| 10 | `card_present` | `1` or `0` |
| 11 | `unknown_merchant` | `1` if the merchant is not in the known list |
| 12 | `mcc_risk` | value from `mcc_risk.json`, default `0.5` |
| 13 | `merchant_avg_amount` | `clamp(merchant.avg_amount / max_merchant_avg_amount)` |

`clamp(x)` limits the value to `[0, 1]`. When there is no previous transaction, dimensions 5 and 6 are set to `-1`, the only case with a value outside the range.

## Search process

The index is a **partitioned KD-tree**:

1. the query is quantized to `int16` with scale `10000`;
2. an 8-bit key selects the partition, with O(1) lookup in `partitionsByKey[256]`;
3. the partition bounding box is compared with the current worst candidate;
4. the tree is traversed with bounds pruning;
5. leaves are scanned with the AVX2 kernel, 8 vectors per block;
6. if the worst distance reaches zero, the search stops immediately.

### Partition key

| Bit | Feature | Condition |
|---|---|---|
| 0 | `last_tx` | `dim[5] >= 0` |
| 1 | `is_online` | `dim[9] > 0` |
| 2 | `card_present` | `dim[10] > 0` |
| 3 | `unknown_merchant` | `dim[11] > 0` |
| 4–5 | `mcc_bucket` | `dim[12]` (0–3) |
| 6 | `amount_high` | `dim[2] > 5000` |
| 7 | `tx_count_high` | `dim[8] > 2500` |

## Allocation-free vectorization

`VectorizeJSON` does not deserialize the entire request. The parser scans the bytes once and extracts only the required fields using `parseFloatFast`, `parseIntFast`, and allocation-free string extraction. Normalization constants are applied through precomputed inverse multiplication, avoiding division on the critical path.

## Quality controls

The choice of exact search was deliberate. An approximate index can select an incorrect neighborhood and produce false positives or false negatives. Because the detection component penalizes errors and is zeroed above a 15% failure rate, the final version prioritized **exactness with aggressive pruning** over approximation.
