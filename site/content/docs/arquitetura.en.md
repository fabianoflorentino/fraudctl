---
title: "Architecture"
description: "Topology, components, and rationale for fraudctl's structural decisions."
weight: 20
---

## Overview

The architecture was designed to preserve predictability and low per-request cost. The main path consists of three elements: a custom TCP load balancer, a byte-oriented HTTP server, and a search index loaded into memory during startup.

<div class="flow">
  <div class="flow-node"><span class="k">client</span><span class="v">Client</span><span class="s">HTTP</span></div>
  <div class="flow-arrow" aria-hidden="true"><svg width="30" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 30 14"><path d="M1 7h26M22 3l5 4-5 4"/></svg></div>
  <div class="flow-node is-accent"><span class="k">load balancer</span><span class="v">fraudctl-lb</span><span class="s">TCP :9999</span></div>
  <div class="flow-arrow" aria-hidden="true"><svg width="30" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 30 14"><path d="M1 7h26M22 3l5 4-5 4"/></svg></div>
  <div class="flow-group">
    <span class="g-label">SCM_RIGHTS</span>
    <div class="g-row">
      <div class="flow-node"><span class="k">api</span><span class="v">api-1</span><span class="s">rawhttp</span></div>
      <div class="flow-node"><span class="k">api</span><span class="v">api-2</span><span class="s">rawhttp</span></div>
    </div>
  </div>
  <div class="flow-arrow" aria-hidden="true"><svg width="30" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 30 14"><path d="M1 7h26M22 3l5 4-5 4"/></svg></div>
  <div class="flow-node"><span class="k">index</span><span class="v">grid.bin</span><span class="s">KD-tree · ~88 MB</span></div>
</div>

## Structural decisions

### Load balancer without an HTTP proxy

`fraudctl-lb` accepts the TCP connection, selects an API in round-robin, and transfers the socket **file descriptor** to the API through a Unix domain socket using `SCM_RIGHTS`. The API rebuilds the connection with `net.FileConn` and processes the request directly.

This decision removes an HTTP proxy layer from the critical path, reducing parsing and data copying. The load balancer remains limited to connection distribution, in compliance with the challenge rules.

```go
// cmd/lb/main.go (simplified excerpt)
rights := syscall.UnixRights(int(fd))
_, _, err := unixConn.WriteMsgUnix(nil, rights, nil)
```

### Purpose-built HTTP server

The `internal/rawhttp` package implements only what the two contract endpoints require:

- byte-oriented request line and header parsing;
- 1 KiB read buffer reused through `sync.Pool`;
- byte-level routing without string allocation;
- complete HTTP responses pre-built in memory;
- 2 KiB request limit;
- `SetNoDelay` and keep-alive on the received connection.

The decision not to use `net/http` or `fasthttp` was based on the cost of a generic API for a fixed, low-complexity contract.

### In-memory index

`grid.bin` is generated at build time from `references.json.gz` and contains:

- 3,000,000 vectors quantized to `int16`;
- partitions defined by an 8-bit key derived from features;
- one KD-tree per partition, with 128-vector leaves;
- bit-packed labels;
- per-partition bounding boxes for pruning.

At startup, the API loads the index and sets the memory limit with `debug.SetMemoryLimit(150 << 20)`. The index is never rebuilt per request.

## Runtime topology

| Service | Limit | Responsibility |
|---|---|---|
| `fraudctl-lb` | 0.04 CPU · 10 MB | TCP acceptance and descriptor distribution |
| `api-1` | 0.48 CPU · 170 MB | Vectorization and KNN search |
| `api-2` | 0.48 CPU · 170 MB | Vectorization and KNN search |

The three services total **1 CPU and 350 MB**, exactly the limit defined by the challenge.
