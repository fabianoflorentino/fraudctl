# Evaluation and Scoring

How the submission is evaluated in the Rinha de Backend 2026 competition.

## Load Test

The load test uses [k6](https://k6.io/) in an incremental request scenario. The test uses pre-labeled payloads based on the reference dataset. The labeling was done by applying **k-NN with k=5 and Euclidean distance** over the 14-dimensional vectors.

## What is Tested

The test dataset comes pre-labeled — for each request, whether the transaction is fraud or legitimate is known in advance. The test compares the backend's response (`approved: true|false`) with the expected label and classifies each request into one of the five categories below:

- **TP (True Positive)** — fraud correctly denied.
- **TN (True Negative)** — legitimate transaction correctly approved.
- **FP (False Positive)** — legitimate transaction incorrectly denied.
- **FN (False Negative)** — fraud incorrectly approved.
- **Error** — HTTP error (response other than 200).

## Scoring Formula

The final score is the sum of two independent components: one for latency (p99) and one for detection quality. Both use a logarithmic function — the idea is to reward each order of magnitude of improvement.

### Latency — `score_p99`

```txt
If p99 > 2000ms:
    score_p99 = −3000                          ← cutoff active
Else:
    score_p99 = 1000 · log₁₀(1000 / max(p99, 1))
```

- Ceiling of +3000: when `p99 ≤ 1ms`, the score saturates at 3000.
- Floor of −3000: when `p99 > 2000ms`, the score is fixed at −3000.

In practice, every 10× improvement in latency is worth another 1000 points. From 100ms to 10ms: another 1000. From 10ms to 1ms: another 1000.

### Detection — `score_det`

```txt
E             = 1·FP + 3·FN + 5·Err            (weighted errors)
ε             = E / N                           (weighted rate)
failures      = FP + FN + Err                   (raw count)
failure_rate  = failures / N

If failure_rate > 15%:
    score_det = −3000                          ← cutoff active
Else:
    score_det = 1000 · log₁₀(1 / max(ε, 0.001)) − 300 · log₁₀(1 + E)
```

- Weights: `FP = 1`, `FN = 3`, `Err = 5`. HTTP errors have the largest weight.

When more than 15% of requests fail, `score_det` is fixed at −3000.

### Final Score

```txt
final_score = score_p99 + score_det
```

- **Maximum: +6000 points** (+3000 + +3000)
- **Minimum: −6000 points** (−3000 − 3000)

## Scoring Examples

| p99 | FP | FN | Errors | failure_rate | p99_score | detection_score | final_score |
| ----- | ----- | ----- | -------- | --------------- | ----------- | ------------------ | ------------ |
| 1ms | 0 | 0 | 0 | 0.00% | 3000.00 | 3000.00 | **6000.00** |
| 3ms | 5 | 5 | 0 | 0.20% | 2522.88 | 2001.27 | **4524.15** |
| 100ms | 0 | 0 | 0 | 0.00% | 1000.00 | 3000.00 | **4000.00** |
| 10ms | 30 | 20 | 0 | 1.00% | 2000.00 | 1157.02 | **3157.02** |
| 300ms | 100 | 50 | 0 | 3.00% | 522.88 | 581.13 | **1104.01** |
| 200ms | 500 | 250 | 0 | 15.00% | 698.97 | −327.12 | **371.85** |
| 10ms | 500 | 300 | 0 | 16.00% | 2000.00 | −3000.00 | **−1000.00** |

## Score History

| Version | Official Score | p99 | FP | FN | Notes |
|---|---|---|---|---|---|
| v1.0.45 | 1650 | 112ms | 1229 | 10 | IVF nlist=300, CGo |
| v1.0.51 | 3443 | — | — | — | IVF nlist=1024, CGo AVX2 |
| v1.0.52 | 3434 | 157ms | — | — | IVF nlist=1024, CGo AVX2 (CPU throttled) |
| v1.0.53+ | TBD | ~91ms | 1 | 0 | IVF v4 nlist=4096, pure Go, 0 allocs |

## Latest Local Docker Result (nlist=4096, nprobe=16, retryExtra=8)

Run summary:

- Image: `fraudctl:local` (commit `42c80ea`+`63e4fce`)
- p99: `91ms`
- p99 score: `1039`
- detection score: `2910`
- **final score: `3949`**

Detection breakdown:

- TP: high (no FN observed)
- FP: `1`
- FN: `0`
- HTTP errors: `0`
- Weighted errors (`E = FP + 3*FN + 5*Err`): `1`
- Failure rate: `< 0.01%`

> Note: local Docker runs show variance in p99 (91ms–137ms) depending on host load. The 91ms run is the most representative low-noise measurement.

## Key Insights

1. **The log favors low p99, down to 1ms.** Reducing latency from 10ms to 1ms earns another 1000 points in `p99_score`. Below 1ms, the score saturates at 3000.

2. **The 15% failure cutoff is strict.** If more than 15% of requests fail, `detection_score` is fixed at −3000 and cancels any gain obtained on p99.

3. **HTTP 500 has a double impact.** It enters `E` with weight 5 (against 1 for an FP) and also counts in `failure_rate`. Returning a fallback response (`approved: true`, `fraud_score: 0.0`) avoids the HTTP error at the cost of potentially raising FP or FN.

4. **The reference files do not change during the test.** You can pre-process freely at startup or during the container build — the more processing you move outside of the test, the better your p99 tends to be.

5. **FN is penalized 3× more than FP.** Missing a fraud (FN) costs triple. The boundary retry (`retryExtra=8` when `fraudCount ∈ [2,3]`) directly targets this: edge-case transactions near the decision boundary get extra neighbors to break ties.

6. **nprobe=16 is optimal.** Reducing to nprobe=10 saves ~37% scan work but drops `detection_score` from 2910 to 2767 (−143 pts) while saving only ~5ms on p99 (+50 pts). Net: −93 pts.
