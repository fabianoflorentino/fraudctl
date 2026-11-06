---
title: "Arquitetura"
description: "Topologia, componentes e justificativa das decisões estruturais do fraudctl."
weight: 20
---

## Visão geral

A arquitetura foi desenhada para preservar previsibilidade e baixo custo por requisição. O caminho principal é composto por três elementos: um balanceador de carga TCP próprio, um servidor HTTP orientado a bytes e um índice de busca carregado em memória durante o startup.

<div class="flow">
  <div class="flow-node"><span class="k">client</span><span class="v">Cliente</span><span class="s">HTTP</span></div>
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

## Decisões estruturais

### Balanceador de carga sem proxy HTTP

O `fraudctl-lb` aceita a conexão TCP, seleciona uma API em round-robin e transfere o **file descriptor** do socket para a API por meio de um Unix domain socket, utilizando `SCM_RIGHTS`. A API reconstrói a conexão com `net.FileConn` e processa a requisição diretamente.

Essa decisão elimina uma camada de proxy HTTP no caminho crítico, reduzindo parsing e cópia de dados. O balanceador permanece restrito à distribuição de conexões, em conformidade com as regras do desafio.

```go
// cmd/lb/main.go (trecho simplificado)
rights := syscall.UnixRights(int(fd))
_, _, err := unixConn.WriteMsgUnix(nil, rights, nil)
```

### Servidor HTTP dedicado

O pacote `internal/rawhttp` implementa apenas o necessário para os dois endpoints do contrato:

- parser de request line e headers orientado a bytes;
- buffer de leitura de 1 KiB reutilizado via `sync.Pool`;
- roteamento por bytes, sem alocação de strings;
- respostas HTTP completas pré-montadas em memória;
- limite de 2 KiB por requisição;
- `SetNoDelay` e keep-alive na conexão recebida.

A decisão de não utilizar `net/http` ou `fasthttp` foi baseada no custo de uma API genérica para um contrato fixo e de baixa complexidade.

### Índice em memória

O `grid.bin` é gerado em tempo de build a partir de `references.json.gz` e contém:

- 3.000.000 de vetores quantizados em `int16`;
- partições definidas por uma chave de 8 bits derivada das features;
- uma KD-tree por partição, com folhas de 128 vetores;
- rótulos bit-packed;
- bounding boxes por partição para poda.

No startup, a API carrega o índice e fixa o limite de memória com `debug.SetMemoryLimit(150 << 20)`. O índice não é reconstruído por requisição.

## Topologia de execução

| Serviço | Limite | Responsabilidade |
|---|---|---|
| `fraudctl-lb` | 0,04 CPU · 10 MB | Aceitação de TCP e distribuição de descritores |
| `api-1` | 0,48 CPU · 170 MB | Vetorização e busca KNN |
| `api-2` | 0,48 CPU · 170 MB | Vetorização e busca KNN |

Os três serviços somam **1 CPU e 350 MB**, exatamente o limite definido pelo desafio.
