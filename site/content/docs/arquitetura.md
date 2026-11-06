---
title: "Arquitetura"
description: "Load balancer custom com SCM_RIGHTS, servidor HTTP orientado a bytes e índice KD-tree em memória."
weight: 20
---

O caminho principal do fraudctl tem três peças: um load balancer TCP próprio, um servidor HTTP escrito à mão e um índice de busca carregado em memória no startup.

<div class="arch-diagram">
  <div class="arch-node"><span class="arch-node-label">Cliente</span><span class="arch-node-code">HTTP</span></div>
  <div class="arch-arrow" aria-hidden="true"><svg width="34" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 34 16"><path d="M1 8h30M25 3l6 5-6 5"/></svg></div>
  <div class="arch-node arch-node-sidecar"><span class="arch-node-label">fraudctl-lb</span><span class="arch-node-code">TCP :9999</span></div>
  <div class="arch-arrow" aria-hidden="true"><svg width="34" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 34 16"><path d="M1 8h30M25 3l6 5-6 5"/></svg></div>
  <div class="arch-pod">
    <span class="arch-pod-title">SCM_RIGHTS</span>
    <div class="arch-pod-inner">
      <div class="arch-node"><span class="arch-node-label">api-1</span><small>rawhttp</small></div>
      <div class="arch-node"><span class="arch-node-label">api-2</span><small>rawhttp</small></div>
    </div>
  </div>
  <div class="arch-arrow" aria-hidden="true"><svg width="34" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 34 16"><path d="M1 8h30M25 3l6 5-6 5"/></svg></div>
  <div class="arch-node"><span class="arch-node-label">grid.bin</span><small>KD-tree · ~88 MB</small><span class="arch-node-code">PredictRaw</span></div>
</div>

## Load balancer sem proxy

O `fraudctl-lb` não faz proxy HTTP. Ele aceita a conexão TCP, escolhe uma API em round-robin e transfere o **file descriptor** do socket para a API através de um Unix domain socket, usando `SCM_RIGHTS` com `WriteMsgUnix`.

No lado da API, `parseUnixRights` lê o descritor do control message, `net.FileConn` reconstrói a conexão e o servidor `rawhttp` processa a requisição diretamente. O resultado é uma camada a menos de parsing e cópia no caminho crítico.

```go
// cmd/lb/main.go (trecho simplificado)
rights := syscall.UnixRights(int(fd))
_, _, err := unixConn.WriteMsgUnix(nil, rights, nil)
```

## Servidor HTTP próprio

`internal/rawhttp` implementa apenas o necessário para dois endpoints:

- parser de request line e headers orientado a bytes;
- buffer de leitura de 1 KiB via `sync.Pool`;
- roteamento por bytes, sem alocar strings;
- respostas HTTP completas pré-montadas em memória;
- limite de 2 KiB por request;
- `SetNoDelay` e keep-alive na conexão recebida via FD passing.

O servidor não usa `net/http` nem `fasthttp`. A decisão foi evitar o custo de uma API genérica para um contrato fixo e minúsculo.

## Índice em memória

O `grid.bin` é gerado em tempo de build a partir de `references.json.gz`. Ele contém:

- 3.000.000 de vetores quantizados em `int16`;
- partições por uma chave de 8 bits derivada das features;
- uma KD-tree por partição, com folhas de 128 vetores;
- labels bit-packed (`fraud`/`legit`);
- bounding boxes por partição para poda.

No startup, a API carrega o índice e fixa o limite de memória com `debug.SetMemoryLimit(150 << 20)`. O índice não é reconstruído por request.

## Topologia do compose

| Serviço | Limite | Papel |
|---|---|---|
| `fraudctl-lb` | 0,04 CPU · 10 MB | Aceita TCP e distribui FDs |
| `api-1` | 0,48 CPU · 170 MB | Vectorize + KNN |
| `api-2` | 0,48 CPU · 170 MB | Vectorize + KNN |

Os três serviços somam **1 CPU e 350 MB**, exatamente o limite da Rinha.
