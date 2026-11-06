---
title: "Results"
description: "Official Rinha de Backend 2026 numbers, score composition, and executive reading of the result."
weight: 60
---

## Official result

The numbers below come from `final-results-2026.json`, published at [rinhadebackend.com.br](https://rinhadebackend.com.br/). The evaluated submission was commit `e90aaa4`, image `fraudctl:v1.0.114`.

| Metric | Value |
|---|---:|
| Rank | **19th / 322** |
| Final score | **5,935.21 / 6,000** |
| p99 | **1.1609 ms** |
| Requests evaluated | 50,000 |
| Edge cases | 5,605 |
| False positives | **0** |
| False negatives | **0** |
| HTTP errors | **0** |

## Score composition

| Component | Points |
|---|---:|
| Latency (p99) | 2,935.21 / 3,000 |
| Detection | 3,000 / 3,000 |
| **Total** | **5,935.21 / 6,000** |

The detection component reached the maximum because there were no decision errors and no HTTP errors. Latency was slightly above 1 ms, the range in which the score saturates at 3,000.

## Executive reading

- **Zero decision errors** across 50,000 requests, including 5,605 edge cases.
- **p99 of 1.16 ms** with 1 CPU and 350 MB in total.
- **19th place** among 322 valid submissions, ahead of solutions with more resources.

Most higher-scoring submissions saturated the latency component (p99 ≤ 1 ms). fraudctl reached 1.16 ms — close enough to secure 2,935 of the 3,000 points, with room for improvement.

## Next steps

With more development time, the natural directions would be:

- reduce p99 below 1 ms;
- measure the real impact of a calibrated prefilter without sacrificing exactness;
- explore parallelism between the two instances more effectively;
- keep the index even closer to L3 cache.
