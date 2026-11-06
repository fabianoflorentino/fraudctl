---
title: "Resultados"
description: "Os números oficiais da Rinha de Backend 2026 e a composição do score do fraudctl."
weight: 60
---

Os números abaixo vêm de `final-results-2026.json`, publicado em [rinhadebackend.com.br](https://rinhadebackend.com.br/). A submissão avaliada foi o commit `e90aaa4`, imagem `fraudctl:v1.0.114`.

## Colocação

| Métrica | Valor |
|---|---:|
| Colocação | **19º / 322** |
| Score final | **5.935,21 / 6.000** |
| p99 | **1,1609 ms** |
| Requisições avaliadas | 50.000 |
| Casos de borda | 5.605 |
| Falsos positivos | **0** |
| Falsos negativos | **0** |
| Erros HTTP | **0** |

## Composição do score

| Componente | Pontos |
|---|---:|
| Latência (p99) | 2.935,21 / 3.000 |
| Detecção | 3.000 / 3.000 |
| **Total** | **5.935,21 / 6.000** |

A componente de detecção foi máxima porque não houve nenhum erro de decisão nem erro HTTP. A latência ficou logo acima de 1 ms, faixa em que o score satura em 3.000.

## O que isso significa

- **Zero erros de decisão** em 50 mil requisições, incluindo 5.605 casos de borda.
- **p99 de 1,16 ms** com 1 CPU e 350 MB no total.
- **19ª colocação** entre 322 submissões válidas, à frente de soluções com mais recursos.

A maior parte dos participantes que ficou acima saturou o score de latência (p99 ≤ 1 ms). O fraudctl chegou a 1,16 ms — perto o suficiente para garantir 2935 dos 3000 pontos, mas ainda com margem para explorar.

## Próximos passos

Com mais tempo, os caminhos naturais seriam:

- reduzir o p99 para abaixo de 1 ms;
- medir o impacto real de um prefilter calibrado sem sacrificar exatidão;
- explorar melhor o paralelismo entre as duas instâncias;
- manter o índice ainda mais próximo da cache L3.
