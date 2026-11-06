---
title: "Lessons"
description: "Decisions kept, decisions discarded, and recommendations for future projects."
weight: 70
---

## Decisions kept

**Exact index with a partitioned KD-tree.** The combination of feature partitions, per-partition KD-tree, bounding-box pruning, and AVX2 kernel reconciled exactness and latency.

**Load balancer with descriptor passing.** Removing the HTTP proxy between the load balancer and the APIs simplified the critical path and reduced overhead without violating the rule that the load balancer must not apply detection logic.

**Purpose-built HTTP server.** For two endpoints and a fixed contract, a byte-oriented parser proved more predictable than `net/http` or `fasthttp`.

**Build-time index generation.** Generating the index during the image build eliminated startup work and focused the runtime on responding to requests.

## Decisions discarded

**Approximate IVF.** It delivered better microbenchmark performance but was approximate. Exact search avoided the error class with the greatest score impact.

**GBDT/RF prefilter.** Tree models were trained and evaluated, but the calibration required to guarantee zero FP/FN did not justify the complexity on the final path.

**Unused fallbacks.** After the event, the repository was cleaned up: dead code, unexercised fallbacks, and model artifacts were removed from the main path.

## Recommendations

- **Measure before optimizing.** Several optimizations only made sense after a CPU profile showed where time was concentrated.
- **Treat p99 as a first-class target.** The difference between 1.16 ms and 1.0 ms represents 65 points in the score.
- **Document decisions during the process.** Retrospective documentation is useful but loses part of the context of each trade-off.
- **Simplify earlier.** Experimental code tends to outlive the experiment; early removal reduces maintenance cost.

## Conclusion

Performance in Go does not depend on micro-optimizing all code. It depends on **identifying the critical path**, eliminating allocations, choosing adequate data structures, and isolating what truly requires assembly. The rest should remain simple, testable, and idiomatic.
