---
title: "Documentação"
description: "Mergulhos técnicos no fraudctl: arquitetura, detecção, performance e as decisões de engenharia por trás da 19ª colocação na Rinha de Backend 2026."
weight: 1
---

O **fraudctl** é uma API de detecção de fraude construída para a Rinha de Backend 2026. O desafio pedia uma busca vetorial exata sobre 3 milhões de referências, com pelo menos um load balancer e duas instâncias da API, tudo dentro de **1 CPU e 350 MB** de memória.

O resultado: **19º lugar entre 322 submissões**, com **p99 de 1,16 ms** e **zero falsos positivos, falsos negativos ou erros HTTP** em 50 mil requisições.

{{< callout >}}
Esta seção é um mergulho no "como" e no "porquê". Cada página parte de uma decisão real do projeto, mostra o código ou o benchmark correspondente e explica o trade-off.
{{< /callout >}}

## Comece por aqui

- **Desafio** — o contrato da Rinha, as restrições e a fórmula de pontuação.
- **Arquitetura** — load balancer custom com `SCM_RIGHTS`, servidor HTTP próprio e índice em memória.
- **Detecção** — as 14 dimensões, a normalização e o KNN exato com KD-tree.
- **Performance** — quantização `int16`, kernel AVX2, zero alocação e PGO.
- **Go idiomático** — como o projeto mantém performance sem abandonar as convenções da linguagem.
- **Resultados** — os números oficiais e a composição do score.
- **Lições** — o que foi testado, o que ficou e o que foi descartado.
