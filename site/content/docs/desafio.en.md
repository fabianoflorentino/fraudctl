---
title: "Challenge"
description: "Scope, functional contract, infrastructure constraints, and evaluation criteria for Rinha de Backend 2026."
weight: 10
---

## Context

Rinha de Backend 2026 proposed building a fraud detection module for card transactions based on vector search. The system must receive transaction data, compare it against a historical dataset of 3 million records, and return an approval decision with a risk score.

The module is consumed by a card authorization system. The challenge scope covers only the detection API, not the full authorization flow.

## Functional contract

| Endpoint | Method | Response |
|---|---|---|
| `/ready` | `GET` | `2xx` when the API is ready |
| `/fraud-score` | `POST` | `200 OK` with `{approved, fraud_score}` |

The payload contains transaction, customer, merchant, and terminal data, plus an optional previous transaction. The complete specification is available in [API.md](https://github.com/zanfranceschi/rinha-de-backend-2026/blob/main/docs/en/API.md).

## Decision rule

1. Turn the payload into a **14-dimensional vector**.
2. Find the **5 nearest vectors** in the reference dataset.
3. Compute `fraud_score` as the fraction of neighbors labeled as fraud.
4. Approve the transaction when `fraud_score < 0.6`.

The score takes only six values — `0.0`, `0.2`, `0.4`, `0.6`, `0.8`, and `1.0` — which allows responses to be precomputed and reduces formatting cost on the critical path.

## Infrastructure constraints

- At least **one load balancer** and **two API instances**, with round-robin distribution.
- The load balancer **must not** apply detection logic.
- `docker-compose.yml` with public `linux-amd64` images.
- **Bridge** network; `host` and `privileged` modes are forbidden.
- Combined resource limits across all services: **1 CPU and 350 MB**.
- Public port: **9999**.
- Test payloads must not be used as reference data or for identifier lookups.

## Evaluation criteria

The final score combines two independent components, each ranging from −3000 to +3000:

- **Latency (`score_p99`)** — each 10× improvement is worth +1000 points; it saturates at +3000 with p99 ≤ 1 ms and is fixed at −3000 above 2000 ms.
- **Detection (`score_det`)** — considers false positives, false negatives, and HTTP errors. Above a 15% failure rate, the score is fixed at −3000.

fraudctl finished the edition with **5,935.21 points** and **zero failures** across 50,000 requests. A detailed reading is available in [Results](../resultados/).
