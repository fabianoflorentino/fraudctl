---
title: "Documentation"
description: "Technical and executive overview of fraudctl: context, architecture, performance, and engineering decisions behind 19th place at Rinha de Backend 2026."
weight: 1
---

## Executive summary

**fraudctl** is a fraud detection module for card transactions, built for Rinha de Backend 2026. The challenge required turning each transaction into a 14-dimensional vector, finding the five most similar historical records in a 3-million-reference dataset, and returning an approval decision in real time, with at least one load balancer and two API instances.

The solution was implemented in Go, with an exact search index built at image build time, and ran under a total budget of **1 CPU and 350 MB** of memory.

**Official result:** 19th place among 322 submissions, with a score of **5,935.21 out of 6,000 points**, a p99 of **1.16 ms**, and no false positives, false negatives, or HTTP errors across 50,000 evaluated requests.

## Purpose of this documentation

This section presents the technical decisions and trade-offs behind the result in language accessible to product leaders, engineering managers, and technical reviewers. Each page combines business context, decision criteria, and implementation evidence.

## Structure

- **Challenge** — scope, functional contract, constraints, and evaluation criteria.
- **Architecture** — topology, components, and rationale for structural choices.
- **Detection** — decision model, dimensions, and search process.
- **Performance** — goals, results, and optimizations with measured impact.
- **Idiomatic Go** — engineering standards, quality, and operations.
- **Results** — official numbers and score interpretation.
- **Lessons** — decisions kept, discarded, and recommendations.

{{< callout >}}
**For management and product:** the official numbers, impact, and score composition are consolidated in [Results](resultados/). The remaining pages cover the "how" and "why" behind each decision.
{{< /callout >}}
